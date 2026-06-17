"""Pydantic v2 config parser — fails fast on malformed YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class ModelSpec(BaseModel):
    id: str
    backend: str = "hf"

    @field_validator("backend")
    @classmethod
    def _valid_backend(cls, v: str) -> str:
        allowed = {"hf", "vllm", "mock"}
        if v not in allowed:
            raise ValueError(f"backend must be one of {allowed}, got '{v}'")
        return v


class DatasetSpec(BaseModel):
    source: str = "hf"
    path: str
    split: str = "test"
    max_samples: int = Field(default=100, gt=0)
    schema_fallback_keys: dict[str, list[str]] = Field(
        default_factory=lambda: {
            "question": ["question", "text", "input", "prompt"],
            "answer": ["answer", "label", "output", "target"],
        }
    )


class CacheConfigSpec(BaseModel):
    """Named cache configuration preset."""
    name: str
    kv_cache: bool
    prefix_cache: bool
    description: str = ""


class HyperparameterSpec(BaseModel):
    quantization: list[Optional[str]] = Field(default_factory=lambda: [None, "int8", "int4"])
    cache_config: list[CacheConfigSpec] = Field(default_factory=list)
    kv_cache: Optional[list[bool]] = None  # Backward compatibility
    prefix_cache: Optional[list[bool]] = None  # Backward compatibility
    max_new_tokens: int | list[int] = Field(default=256)

    def _normalize_max_tokens(self) -> list[int]:
        """Normalize max_new_tokens to always be a list."""
        if isinstance(self.max_new_tokens, list):
            return self.max_new_tokens
        return [self.max_new_tokens]

    def _get_cache_configs(self) -> list[dict[str, bool]]:
        """Extract cache configs, supporting both old and new formats."""
        if self.cache_config:
            return [{"kv_cache": c.kv_cache, "prefix_cache": c.prefix_cache} for c in self.cache_config]
        # Fallback to old format for backward compatibility
        if self.kv_cache and self.prefix_cache:
            configs = []
            for kv in self.kv_cache:
                for prefix in self.prefix_cache:
                    configs.append({"kv_cache": kv, "prefix_cache": prefix})
            return configs
        return [{"kv_cache": False, "prefix_cache": False}]


class GenerationSpec(BaseModel):
    temperature: float = 0.0
    top_p: float = 1.0
    repetition_penalty: float = 1.0


class HardwareSpec(BaseModel):
    monitor_interval_ms: int = Field(default=50, ge=10)
    gpu_device_index: int = 0
    tdp_limit_watts: int = 300


class OutputSpec(BaseModel):
    results_dir: str = "results"
    compress: str = "snappy"

    @field_validator("compress")
    @classmethod
    def _valid_compress(cls, v: str) -> str:
        allowed = {"snappy", "gzip", "brotli", "zstd", "none"}
        if v not in allowed:
            raise ValueError(f"compress must be one of {allowed}")
        return v


class MockSpec(BaseModel):
    enabled: bool = False
    simulate_oom_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    base_latency_ms: int = Field(default=120, gt=0)
    tokens_per_second: int = Field(default=80, gt=0)


class ExperimentSpec(BaseModel):
    name: str
    description: str = ""
    seed: int = 42


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------

class BenchmarkConfig(BaseModel):
    experiment: ExperimentSpec
    models: list[ModelSpec]
    dataset: DatasetSpec
    hyperparameters: HyperparameterSpec = Field(default_factory=HyperparameterSpec)
    generation: GenerationSpec = Field(default_factory=GenerationSpec)
    hardware: HardwareSpec = Field(default_factory=HardwareSpec)
    output: OutputSpec = Field(default_factory=OutputSpec)
    mock: MockSpec = Field(default_factory=MockSpec)

    @model_validator(mode="after")
    def _mock_overrides_backends(self) -> "BenchmarkConfig":
        if self.mock.enabled:
            for m in self.models:
                if m.backend != "mock":
                    m.backend = "mock"
        return self

    def permutations(self) -> list[dict[str, Any]]:
        """Return all hyperparameter permutation dicts."""
        from itertools import product

        hp = self.hyperparameters
        cache_configs = hp._get_cache_configs()
        max_tokens_list = hp._normalize_max_tokens()

        combos = product(
            hp.quantization,
            cache_configs,
            max_tokens_list,
        )

        results = []
        for quant, cache_cfg, max_tok in combos:
            results.append({
                "quantization": quant,
                "kv_cache": cache_cfg["kv_cache"],
                "prefix_cache": cache_cfg["prefix_cache"],
                "max_new_tokens": max_tok,
            })
        return results


def load_config(path: str | Path) -> BenchmarkConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    raw = yaml.safe_load(path.read_text())
    return BenchmarkConfig.model_validate(raw)
