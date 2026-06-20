# LLM Local Benchmarking Matrix

Benchmarking suite for sweeping LLM hyperparameter permutations (quantization,
KV-caching, prefix-caching) across local GPU backends (HuggingFace Transformers,
vLLM) and API backends (OpenAI, OpenRouter, Gemini, Maritaca, or any
OpenAI-API-compatible provider). Each permutation runs in an isolated subprocess
so GPU telemetry (VRAM, power draw, utilization) is never cross-contaminated,
and results are checkpointed to Parquet after every cell.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

vLLM on Blackwell (RTX 6000 / sm_100) may need a source build or custom wheel —
see the comment above the `vllm` line in `requirements.txt`.

If you'll use an API backend or API judge, export the relevant key before running:

```bash
export OPENAI_API_KEY=sk-...
export OPENROUTER_API_KEY=sk-or-...
export GEMINI_API_KEY=...
export MARITACA_API_KEY=...
```

## Running a sweep

```bash
python3 run_sweep.py --config config/matrix_search.yaml
```

- `--config PATH` — which YAML to run (default `config/matrix_search.yaml`)
- `--mock` — force every model to the no-GPU, no-network mock backend regardless
  of what the config says. Useful for validating a config or the pipeline itself
  before spending GPU time or API credits.

Each run prints progress per cell and writes incrementally to:

```
results/<dir-from-config>/sweep_<experiment-name>_<UTC-timestamp>.parquet
```

The file is rewritten after every cell completes, so a crash mid-sweep still
leaves you with all completed cells on disk — nothing is lost except the cell
in flight.

### Quick validation run (no GPU, no API calls, no cost)

```bash
python3 run_sweep.py --config config/matrix_search.yaml --mock
```

This swaps in `MockInferenceEngine`, which simulates latency/tokens-per-second
without touching a real model. Use it to confirm a config parses and the
dataset/prompt-building pipeline works before running the real thing.

## Configuring a sweep

A config file has these top-level sections — see `config/matrix_search.yaml`
and `config/matrix_search_diffusiongemma.yaml` for full working examples:

```yaml
experiment:
  name: "my-sweep"          # used in output filenames
  seed: 42

models:
  - id: "google/gemma-4-31B-it-qat-w4a16-ct"
    backend: "vllm"          # "hf" | "vllm" | "api" | "mock"
    max_seq_length: 4096     # vLLM max_model_len; optional

dataset:
  source: "hf"
  path: "maritaca-ai/enem"   # any HF dataset
  split: "train"
  max_samples: 10            # cap rows pulled from the dataset

hyperparameters:
  quantization: [null, "int4"]        # sweep axis
  cache_config:                        # sweep axis
    - name: "baseline"
      kv_cache: false
      prefix_cache: false
    - name: "kv_only"
      kv_cache: true
      prefix_cache: false
  max_new_tokens: 256                  # scalar or list — also a sweep axis

generation:
  temperature: 0.0
  top_p: 1.0
  repetition_penalty: 1.0

output:
  results_dir: "results/tier1"
  compress: "snappy"                   # snappy | gzip | brotli | zstd | none
```

The total number of cells run is `len(models) × product(len of each
hyperparameter axis)`. With the example above: 1 model × 2 quantizations × 2
cache configs × 1 token-length = 4 cells.

**The config schema rejects unknown keys at load time** (fail-fast validation,
not silent ignoring) — if you typo a field name or add a section that isn't
backed by code, `run_sweep.py` will refuse to start and tell you exactly which
key is the problem, rather than quietly running with that option doing nothing.

### Choosing a backend

| `backend` | What it does | Needs |
|---|---|---|
| `vllm` | Local inference via vLLM | GPU, model weights (downloaded on first use) |
| `hf` | Local inference via 🤗 Transformers, supports `int8`/`int4` bitsandbytes quantization | GPU |
| `api` | Calls a hosted model over an OpenAI-compatible `/chat/completions` endpoint | `api_config` block + the matching env var |
| `mock` | Simulated latency, no GPU/network | nothing |

For `backend: "api"`, add an `api_config` to the model:

```yaml
models:
  - id: "gpt-4o-mini"
    backend: "api"
    api_config:
      provider: "openai"   # "openai" | "openrouter" | "gemini" | "maritaca" | "custom"
      # model: "..."        # override if the API-side name differs from `id`
      # base_url: "..."     # required if provider: "custom"
```

Known providers (`openai`, `openrouter`, `gemini`, `maritaca`) resolve their
own `base_url` and the env var they read the API key from automatically — you
only need to export the key, not configure the URL.

### Using a real LLM-as-judge instead of the built-in heuristic

By default no judge runs. To score predictions with a real model:

```yaml
judge:
  enabled: true
  provider: "api"            # "mock" (offline heuristic) | "api" (real LLM call)
  api_config:
    provider: "openrouter"
    model: "anthropic/claude-3.5-sonnet"
```

### vLLM-specific tuning

```yaml
models:
  - id: "nvidia/diffusiongemma-26B-A4B-IT-NVFP4"
    backend: "vllm"
    vllm_advanced:
      trust_remote_code: true
      max_num_seqs: 4
```

Note: vLLM's `serve` CLI exposes flags (`--tool-call-parser`,
`--reasoning-parser`, etc.) that don't exist on the Python `LLM()` API used
here — those aren't configurable through this tool.

## Reading results

Results are Parquet files with one row per cell: hyperparameter config,
accuracy (`exact_match_pct`, `normalized_match_pct`, `judge_avg_score`),
throughput/latency, hardware telemetry (peak/avg VRAM, power, GPU util), and
the full list of prompts/references/predictions for that cell.

Use `extract_results.py` rather than loading the Parquet by hand:

```bash
# Human-readable summary + per-cell breakdown of the most recent sweep
python3 extract_results.py

# Same, but for a specific sweep / output directory
python3 extract_results.py --file results/tier1/sweep_xxx.parquet
python3 extract_results.py --dir results/diffusiongemma

# List available sweep files in a directory
python3 extract_results.py --list

# Export the metrics table to CSV
python3 extract_results.py --export-csv results/summary.csv

# Dump every prompt/reference/prediction per cell to its own CSV
python3 extract_results.py --export-predictions results/predictions/

# Spot-check actual model outputs from one cell
python3 extract_results.py --show-predictions 0 -n 10
```

## Development

```bash
pip install ruff ty           # pytest is already in requirements.txt
ruff check .                  # lint
ty check .                    # type check
pytest tests/ -v              # unit tests (config validation, scoring logic)
```

## Project layout

```
config/             Sweep definitions (YAML)
prompts/             Prompt templates with schema-tolerant dataset field mapping
src/
  config_parser.py   Pydantic schema — fails fast on unknown/invalid keys
  inference_engine.py  HF / vLLM / OpenAI-compatible / Mock backends
  evaluator.py       Exact-match, normalized-match, and LLM-as-judge scoring
  hardware_monitor.py  Background NVML/psutil telemetry sampler
  orchestrator.py    Runs each sweep cell in an isolated subprocess, checkpoints to Parquet
run_sweep.py         CLI entry point
extract_results.py   CLI for reading/exporting Parquet results
tests/                pytest suite
```
