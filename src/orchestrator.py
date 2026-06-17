"""
Sequential sweep orchestrator using isolated sub-processes per permutation.
Each worker runs independently so GPU telemetry is never cross-contaminated.
"""

from __future__ import annotations

import dataclasses
import logging
import multiprocessing as mp
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .config_parser import BenchmarkConfig
from .evaluator import Evaluator
from .hardware_monitor import HardwareMonitor
from .inference_engine import build_engine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Worker entry-point (runs in isolated sub-process)
# ---------------------------------------------------------------------------

def _worker(
    result_queue: "mp.Queue[dict]",
    model_id: str,
    backend: str,
    hyperparams: dict,
    prompts: list[str],
    references: list[str],
    generation_cfg: dict,
    mock_cfg_dict: dict,
    use_judge: bool,
) -> None:
    """Full benchmark run for one (model, hyperparams) cell. Sends result dict to queue."""
    logging.basicConfig(level=logging.INFO)

    from .config_parser import MockSpec
    from .evaluator import Evaluator
    from .hardware_monitor import HardwareMonitor
    from .inference_engine import build_engine

    result: dict[str, Any] = {
        "model_id": model_id,
        "backend": backend,
        **hyperparams,
        "status": "pending",
        "error": None,
    }

    mock_spec = MockSpec(**mock_cfg_dict)

    try:
        engine = build_engine(backend, mock_spec if backend == "mock" else None)
        monitor = HardwareMonitor(interval_ms=50)
        evaluator = Evaluator(use_judge=use_judge)

        predictions: list[str] = []
        ttfts: list[float] = []
        tps_list: list[float] = []
        total_s_list: list[float] = []

        with monitor:
            engine.load_model(model_id, hyperparams)

            for prompt in prompts:
                text, meta = engine.generate(
                    prompt,
                    max_new_tokens=hyperparams.get("max_new_tokens", 256),
                    **generation_cfg,
                )
                predictions.append(text)
                if meta.get("ttft_s") is not None:
                    ttfts.append(meta["ttft_s"])
                tps_list.append(meta["tps"])
                total_s_list.append(meta["total_s"])

            engine.cleanup()

        hw = monitor.summarize()
        eval_summary = evaluator.evaluate_batch(prompts, references, predictions)

        n = len(predictions)
        result.update({
            "status": "ok",
            "n_samples": n,
            # throughput / latency
            "mean_ttft_s":  sum(ttfts) / len(ttfts) if ttfts else None,
            "mean_tps":     sum(tps_list) / n if n else 0.0,
            "mean_total_s": sum(total_s_list) / n if n else 0.0,
            # accuracy
            "exact_match_pct":      eval_summary.exact_match_pct,
            "normalized_match_pct": eval_summary.normalized_match_pct,
            "judge_avg_score":      eval_summary.judge_avg_score,
            # hardware
            "peak_vram_mb":      hw.peak_vram_mb,
            "avg_vram_mb":       hw.avg_vram_mb,
            "peak_power_w":      hw.peak_power_w,
            "avg_power_w":       hw.avg_power_w,
            "avg_gpu_util_pct":  hw.avg_gpu_util_pct,
            "avg_cpu_util_pct":  hw.avg_cpu_util_pct,
            "hw_sample_count":   hw.sample_count,
            "gpu_available":     hw.gpu_available,
            # dataset & predictions (for debugging & analysis)
            "prompts":      prompts,
            "references":   references,
            "predictions":  predictions,
        })

    except RuntimeError as e:
        msg = str(e)
        if "out of memory" in msg.lower() or "cuda" in msg.lower():
            result.update({"status": "oom", "error": msg})
            logger.error("OOM on %s %s: %s", model_id, hyperparams, msg)
        else:
            result.update({"status": "error", "error": msg})
            logger.error("Runtime error on %s %s: %s", model_id, hyperparams, msg)
    except Exception as e:
        result.update({"status": "error", "error": traceback.format_exc()})
        logger.error("Unexpected error on %s %s: %s", model_id, hyperparams, e)

    result_queue.put(result)


# ---------------------------------------------------------------------------
# Cache probing helpers
# ---------------------------------------------------------------------------

def _dataset_is_cached(path: str, split: str) -> bool:
    try:
        from datasets import load_dataset
        load_dataset(path, split=split, local_files_only=True)
        return True
    except Exception:
        return False


def _model_is_cached(model_id: str) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache, _CACHED_NO_EXIST
        result = try_to_load_from_cache(model_id, filename="config.json")
        return result is not None and result is not _CACHED_NO_EXIST
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Dataset loader (moved outside sub-process for efficiency)
# ---------------------------------------------------------------------------

def _load_dataset(cfg: BenchmarkConfig) -> tuple[list[str], list[str]]:
    from datasets import load_dataset
    from prompts.templates import build_prompt, extract_reference_answer

    cached = _dataset_is_cached(cfg.dataset.path, cfg.dataset.split)
    logger.info(
        "Dataset '%s' (split=%s) — %s",
        cfg.dataset.path, cfg.dataset.split,
        "found in local cache" if cached else "not cached, will download",
    )

    ds = load_dataset(
        cfg.dataset.path,
        split=cfg.dataset.split,
    )
    ds = ds.select(range(min(cfg.dataset.max_samples, len(ds))))

    prompts, references = [], []
    for record in ds:
        try:
            prompts.append(build_prompt(dict(record)))
            references.append(extract_reference_answer(dict(record)))
        except KeyError as e:
            logger.warning("Skipping record with missing key: %s", e)

    logger.info("Loaded %d samples.", len(prompts))
    return prompts, references


# ---------------------------------------------------------------------------
# Parquet sink — incremental upsert (rewrites all rows collected so far)
# ---------------------------------------------------------------------------

def _upsert_parquet(records: list[dict], path: Path, compression: str | None) -> None:
    """Overwrite the file with all records collected so far."""
    df = pd.DataFrame(records)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, path, compression=compression)
    logger.info("Checkpoint → %s  (%d rows)", path, len(records))


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class Orchestrator:
    def __init__(self, cfg: BenchmarkConfig):
        self.cfg = cfg
        self._mp_ctx = mp.get_context("spawn")

    def run(self) -> Path:
        cfg = self.cfg
        experiment_token = cfg.experiment.name.replace(" ", "_").lower()
        compression = cfg.output.compress if cfg.output.compress != "none" else None

        # Determine output path once so all incremental writes go to the same file.
        results_dir = Path(cfg.output.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = results_dir / f"sweep_{experiment_token}_{ts}.parquet"

        prompts, references = _load_dataset(cfg)
        permutations = cfg.permutations()
        total_cells = len(cfg.models) * len(permutations)

        logger.info(
            "Starting sweep: %d models × %d hyperparameter combos = %d cells → %s",
            len(cfg.models), len(permutations), total_cells, out_path,
        )

        all_records: list[dict] = []
        cell_idx = 0

        generation_cfg = {
            "temperature": cfg.generation.temperature,
            "top_p": cfg.generation.top_p,
            "repetition_penalty": cfg.generation.repetition_penalty,
        }
        mock_cfg_dict = cfg.mock.model_dump()

        for model_spec in cfg.models:
            cached = _model_is_cached(model_spec.id)
            logger.info(
                "Model '%s' — %s",
                model_spec.id,
                "found in local cache" if cached else "not cached, will download on first cell",
            )

            for hp in permutations:
                cell_idx += 1
                logger.info(
                    "[%d/%d] model=%s backend=%s hp=%s",
                    cell_idx, total_cells, model_spec.id, model_spec.backend, hp,
                )

                result = self._run_cell_isolated(
                    model_id=model_spec.id,
                    backend=model_spec.backend,
                    hyperparams=hp,
                    prompts=prompts,
                    references=references,
                    generation_cfg=generation_cfg,
                    mock_cfg_dict=mock_cfg_dict,
                )
                result["cell_idx"] = cell_idx
                result["experiment"] = cfg.experiment.name
                all_records.append(result)

                status = result["status"]
                if status == "oom":
                    logger.warning("OOM — skipping to next permutation.")
                elif status == "error":
                    logger.error("Cell failed: %s", result.get("error", ""))

                # Persist immediately after every cell.
                _upsert_parquet(all_records, out_path, compression)

        logger.info(
            "Sweep complete. %d/%d cells succeeded.",
            sum(1 for r in all_records if r["status"] == "ok"), total_cells,
        )
        return out_path

    def _run_cell_isolated(
        self,
        model_id: str,
        backend: str,
        hyperparams: dict,
        prompts: list[str],
        references: list[str],
        generation_cfg: dict,
        mock_cfg_dict: dict,
    ) -> dict:
        """Spawn a fresh sub-process for each (model, hyperparams) cell."""
        queue: mp.Queue = self._mp_ctx.Queue()

        proc = self._mp_ctx.Process(
            target=_worker,
            args=(
                queue,
                model_id,
                backend,
                hyperparams,
                prompts,
                references,
                generation_cfg,
                mock_cfg_dict,
                False,   # use_judge: set True to enable Sabiá-4 judge per-call
            ),
            daemon=False,
        )

        proc.start()
        proc.join(timeout=3600)   # 1-hour hard timeout per cell

        # Check for process exit failure
        if proc.exitcode is None:
            return {
                "model_id": model_id,
                "backend": backend,
                **hyperparams,
                "status": "error",
                "error": "Worker process timeout: did not complete within 3600s",
            }

        if proc.exitcode != 0:
            try:
                result = queue.get_nowait()
                return result
            except:
                return {
                    "model_id": model_id,
                    "backend": backend,
                    **hyperparams,
                    "status": "error",
                    "error": f"Worker exited with code {proc.exitcode}",
                }

        # Process exited successfully (exitcode=0)
        try:
            result = queue.get_nowait()
            return result
        except Exception as e:
            return {
                "model_id": model_id,
                "backend": backend,
                **hyperparams,
                "status": "error",
                "error": f"Worker produced no output: {e}",
            }
