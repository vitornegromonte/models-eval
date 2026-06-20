"""Pydantic v2 config parser — fails fast on malformed YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    """Base for all config models: unknown keys fail validation immediately
    instead of being silently dropped. Without this, a typo'd or aspirational
    YAML key (e.g. a config section that was never wired into the orchestrator)
    parses successfully and silently does nothing — exactly the failure mode
    this parser's docstring promises to prevent."""
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class VLLMAdvancedSpec(StrictModel):
    """Advanced vLLM Python API options for specialized models (DiffusionGemma, etc.)"""
    trust_remote_code: bool = False
    attention_backend: Optional[str] = None  # e.g., "triton", "flash_attn", "xformers"
    max_num_seqs: Optional[int] = None
    # Note: tool_call_parser, reasoning_parser, etc. are vLLM serve CLI args, not Python API


class APISpec(StrictModel):
    """
    OpenAI-API-compatible client config. Works unmodified with OpenAI,
    OpenRouter (proxies Claude, Gemini, Llama, etc.), and Gemini's native
    OpenAI-compat endpoint — anything implementing /chat/completions.
    Set `base_url`/`api_key_env` explicitly for self-hosted or unlisted providers.
    """
    provider: str = "openai"            # "openai" | "openrouter" | "gemini" | "maritaca" | "custom"
    base_url: Optional[str] = None      # overrides the provider preset below
    model: Optional[str] = None         # API-side model name; defaults to ModelSpec.id
    api_key_env: Optional[str] = None   # env var holding the key; defaults to provider preset
    timeout_s: float = 60.0
    max_retries: int = 2
    extra_headers: dict[str, str] = Field(default_factory=dict)

    _PROVIDER_BASE_URLS: ClassVar[dict[str, str]] = {
        "openai":     "https://api.openai.com/v1",
        "openrouter": "https://openrouter.ai/api/v1",
        "gemini":     "https://generativelanguage.googleapis.com/v1beta/openai/",
        # Maritaca's Sabiá models speak the same chat/completions schema —
        # no bespoke client needed, just this preset. Verify against current
        # Maritaca docs if requests start failing (endpoints occasionally move).
        "maritaca":   "https://chat.maritaca.ai/api",
    }
    _PROVIDER_KEY_ENVS: ClassVar[dict[str, str]] = {
        "openai":     "OPENAI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "gemini":     "GEMINI_API_KEY",
        "maritaca":   "MARITACA_API_KEY",
    }

    def resolved_base_url(self) -> Optional[str]:
        return self.base_url or self._PROVIDER_BASE_URLS.get(self.provider)

    def resolved_key_env(self) -> str:
        return self.api_key_env or self._PROVIDER_KEY_ENVS.get(self.provider, "OPENAI_API_KEY")


class ModelSpec(StrictModel):
    id: str
    backend: str = "hf"
    description: str = ""  # informational only, shown in logs
    max_seq_length: Optional[int] = None  # vLLM: max_model_len; HF: not yet wired
    vllm_advanced: Optional[VLLMAdvancedSpec] = None
    api_config: Optional[APISpec] = None  # required when backend == "api"

    @field_validator("backend")
    @classmethod
    def _valid_backend(cls, v: str) -> str:
        allowed = {"hf", "vllm", "mock", "api"}
        if v not in allowed:
            raise ValueError(f"backend must be one of {allowed}, got '{v}'")
        return v


class DatasetSpec(StrictModel):
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


class CacheConfigSpec(StrictModel):
    """Named cache configuration preset."""
    name: str
    kv_cache: bool
    prefix_cache: bool
    description: str = ""


class HyperparameterSpec(StrictModel):
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


class GenerationSpec(StrictModel):
    temperature: float = 0.0
    top_p: float = 1.0
    repetition_penalty: float = 1.0


class HardwareSpec(StrictModel):
    monitor_interval_ms: int = Field(default=50, ge=10)
    gpu_device_index: int = 0
    tdp_limit_watts: int = 300


class OutputSpec(StrictModel):
    results_dir: str = "results"
    compress: str = "snappy"

    @field_validator("compress")
    @classmethod
    def _valid_compress(cls, v: str) -> str:
        allowed = {"snappy", "gzip", "brotli", "zstd", "none"}
        if v not in allowed:
            raise ValueError(f"compress must be one of {allowed}")
        return v


class MockSpec(StrictModel):
    enabled: bool = False
    simulate_oom_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    base_latency_ms: int = Field(default=120, gt=0)
    tokens_per_second: int = Field(default=80, gt=0)


class ExperimentSpec(StrictModel):
    name: str
    description: str = ""
    seed: int = 42


class JudgeSpec(StrictModel):
    """LLM-as-a-judge configuration. 'mock' keeps the existing deterministic
    heuristic judge; 'api' calls a real OpenAI-compatible chat model as judge."""
    enabled: bool = False
    provider: str = "mock"  # "mock" | "api"
    api_config: Optional[APISpec] = None

    @field_validator("provider")
    @classmethod
    def _valid_provider(cls, v: str) -> str:
        allowed = {"mock", "api"}
        if v not in allowed:
            raise ValueError(f"judge.provider must be one of {allowed}, got '{v}'")
        return v

    @model_validator(mode="after")
    def _api_requires_config(self) -> "JudgeSpec":
        if self.provider == "api" and self.api_config is None:
            raise ValueError("judge.provider='api' requires judge.api_config to be set")
        return self


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------

class BenchmarkConfig(StrictModel):
    experiment: ExperimentSpec
    models: list[ModelSpec]
    dataset: DatasetSpec
    hyperparameters: HyperparameterSpec = Field(default_factory=HyperparameterSpec)
    generation: GenerationSpec = Field(default_factory=GenerationSpec)
    hardware: HardwareSpec = Field(default_factory=HardwareSpec)
    output: OutputSpec = Field(default_factory=OutputSpec)
    mock: MockSpec = Field(default_factory=MockSpec)
    judge: JudgeSpec = Field(default_factory=JudgeSpec)

    @model_validator(mode="after")
    def _mock_overrides_backends(self) -> "BenchmarkConfig":
        if self.mock.enabled:
            for m in self.models:
                if m.backend != "mock":
                    m.backend = "mock"
        return self

    @model_validator(mode="after")
    def _api_backend_requires_config(self) -> "BenchmarkConfig":
        for m in self.models:
            if m.backend == "api" and m.api_config is None:
                raise ValueError(f"model '{m.id}' has backend='api' but no api_config set")
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
