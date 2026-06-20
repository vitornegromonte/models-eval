"""Config validation must fail fast — these tests exist because that guarantee
silently broke once before (unknown YAML keys were dropped instead of rejected)."""

import pytest
from pydantic import ValidationError

from src.config_parser import APISpec, BenchmarkConfig


def _minimal_config(**overrides: dict) -> dict:
    base = {
        "experiment": {"name": "test"},
        "models": [{"id": "m", "backend": "mock"}],
        "dataset": {"path": "some/dataset"},
    }
    base.update(overrides)
    return base


def test_minimal_config_loads():
    cfg = BenchmarkConfig.model_validate(_minimal_config())
    assert cfg.models[0].backend == "mock"


def test_unknown_top_level_key_is_rejected():
    with pytest.raises(ValidationError):
        BenchmarkConfig.model_validate(_minimal_config(execution={"mode": "distributed"}))


def test_unknown_nested_key_is_rejected():
    cfg = _minimal_config()
    cfg["generation"] = {"temperature": 0.0, "do_sample": False}
    with pytest.raises(ValidationError):
        BenchmarkConfig.model_validate(cfg)


def test_unknown_model_key_is_rejected():
    cfg = _minimal_config()
    cfg["models"] = [{"id": "m", "backend": "mock", "totally_made_up_field": 1}]
    with pytest.raises(ValidationError):
        BenchmarkConfig.model_validate(cfg)


def test_api_backend_requires_api_config():
    cfg = _minimal_config()
    cfg["models"] = [{"id": "gpt-4o-mini", "backend": "api"}]
    with pytest.raises(ValidationError, match="api_config"):
        BenchmarkConfig.model_validate(cfg)


def test_api_backend_with_config_is_accepted():
    cfg = _minimal_config()
    cfg["models"] = [{"id": "gpt-4o-mini", "backend": "api", "api_config": {"provider": "openai"}}]
    parsed = BenchmarkConfig.model_validate(cfg)
    assert parsed.models[0].api_config.provider == "openai"


def test_judge_api_provider_requires_api_config():
    cfg = _minimal_config(judge={"enabled": True, "provider": "api"})
    with pytest.raises(ValidationError, match="api_config"):
        BenchmarkConfig.model_validate(cfg)


def test_invalid_backend_rejected():
    cfg = _minimal_config()
    cfg["models"] = [{"id": "m", "backend": "not-a-real-backend"}]
    with pytest.raises(ValidationError):
        BenchmarkConfig.model_validate(cfg)


def test_mock_enabled_forces_all_models_to_mock_backend():
    cfg = _minimal_config(mock={"enabled": True})
    cfg["models"] = [{"id": "m", "backend": "vllm"}]
    parsed = BenchmarkConfig.model_validate(cfg)
    assert parsed.models[0].backend == "mock"


class TestAPISpecProviderPresets:
    def test_openai_preset(self):
        spec = APISpec(provider="openai")
        assert spec.resolved_base_url() == "https://api.openai.com/v1"
        assert spec.resolved_key_env() == "OPENAI_API_KEY"

    def test_openrouter_preset(self):
        spec = APISpec(provider="openrouter")
        assert spec.resolved_base_url() == "https://openrouter.ai/api/v1"
        assert spec.resolved_key_env() == "OPENROUTER_API_KEY"

    def test_maritaca_preset(self):
        # Regression test: this preset previously didn't exist, and direct
        # dict-based construction (bypassing the orchestrator's eager resolution
        # helper) silently fell back to OPENAI_API_KEY instead of MARITACA_API_KEY.
        spec = APISpec(provider="maritaca")
        assert spec.resolved_key_env() == "MARITACA_API_KEY"
        assert spec.resolved_base_url() is not None

    def test_explicit_base_url_overrides_preset(self):
        spec = APISpec(provider="openai", base_url="https://custom.example.com/v1")
        assert spec.resolved_base_url() == "https://custom.example.com/v1"

    def test_explicit_key_env_overrides_preset(self):
        spec = APISpec(provider="openai", api_key_env="MY_CUSTOM_KEY")
        assert spec.resolved_key_env() == "MY_CUSTOM_KEY"

    def test_unknown_provider_falls_back_to_openai_key_env(self):
        spec = APISpec(provider="custom")
        assert spec.resolved_key_env() == "OPENAI_API_KEY"
        assert spec.resolved_base_url() is None


def test_permutations_cover_full_cross_product():
    cfg = BenchmarkConfig.model_validate(_minimal_config(hyperparameters={
        "quantization": [None, "int4"],
        "cache_config": [
            {"name": "a", "kv_cache": False, "prefix_cache": False},
            {"name": "b", "kv_cache": True, "prefix_cache": False},
        ],
        "max_new_tokens": [128, 256],
    }))
    perms = cfg.permutations()
    assert len(perms) == 2 * 2 * 2
    assert {"quantization": None, "kv_cache": False, "prefix_cache": False, "max_new_tokens": 128} in perms
