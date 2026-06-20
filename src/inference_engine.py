"""Inference engine abstraction: HF Transformers, vLLM, OpenAI-API-compatible, and Mock backends."""

from __future__ import annotations

import gc
import logging
import os
import random
import time
from abc import ABC, abstractmethod
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseInferenceEngine(ABC):
    def __enter__(self) -> "BaseInferenceEngine":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.cleanup()

    @abstractmethod
    def load_model(self, model_id: str, config: dict) -> None: ...

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> tuple[str, dict]: ...

    @abstractmethod
    def cleanup(self) -> None: ...


# ---------------------------------------------------------------------------
# HuggingFace Transformers backend
# ---------------------------------------------------------------------------

class HFInferenceEngine(BaseInferenceEngine):
    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._model_id: Optional[str] = None

    def load_model(self, model_id: str, config: dict) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self._model_id = model_id
        quant = config.get("quantization")

        bnb_config = None
        if quant == "int8":
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        elif quant == "int4":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        logger.info("Loading %s (quant=%s) via HF Transformers …", model_id, quant)
        self._tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16 if bnb_config is None else None,
            device_map="auto",
        )
        self._model.eval()

    def generate(self, prompt: str, **kwargs) -> tuple[str, dict]:
        import torch

        max_new_tokens = kwargs.get("max_new_tokens", 256)

        # Instruction-tuned models (Gemma-4, etc.) require the chat template's
        # turn-delimiter tokens; feeding raw text causes repetition-loop degeneration.
        chat_prompt = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._tokenizer(chat_prompt, return_tensors="pt", add_special_tokens=False).to(self._model.device)

        t0 = time.perf_counter()
        with torch.inference_mode():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=kwargs.get("temperature", 0.0),
                do_sample=kwargs.get("temperature", 0.0) > 0,
                repetition_penalty=kwargs.get("repetition_penalty", 1.0),
                pad_token_id=self._tokenizer.eos_token_id,
            )
        elapsed = time.perf_counter() - t0

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
        n_tokens = len(generated_ids)

        meta = {
            "ttft_s": None,        # not measurable with batch HF generate
            "tps": n_tokens / elapsed if elapsed > 0 else 0.0,
            "total_s": elapsed,
            "n_tokens": n_tokens,
        }
        return text.strip(), meta

    def cleanup(self) -> None:
        try:
            import torch
            del self._model
            del self._tokenizer
            self._model = self._tokenizer = None
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            # Not debug: a failed cleanup means GPU memory likely wasn't freed,
            # which can starve the next cell in the sweep — this should be visible.
            logger.warning("HF cleanup error (GPU memory may not be freed): %s", e)


# ---------------------------------------------------------------------------
# vLLM backend
# ---------------------------------------------------------------------------

class VLLMInferenceEngine(BaseInferenceEngine):

    def __init__(self) -> None:
        self._llm = None
        self._sampling_params = None

    def load_model(self, model_id: str, config: dict) -> None:
        try:
            from vllm import LLM
        except ImportError as e:
            raise RuntimeError(
                "vLLM is not installed. For Blackwell (sm_100) you may need to compile from "
                "source or supply a custom wheel. See requirements.txt."
            ) from e

        quant = config.get("quantization")

        # Map quantization types to vLLM quantization methods
        # Note: Google's Gemma-4-E4B is pre-quantized; we use bitsandbytes for on-the-fly quantization
        quant_map = {
            None: None,                    # Full precision
            "int8": "bitsandbytes",        # On-the-fly int8 quantization
            "int4": "bitsandbytes",        # On-the-fly int4 quantization (NF4)
        }

        # Determine quantization method
        quant_method = quant_map.get(quant)

        logger.info(
            "Loading %s via vLLM (quantization=%s, method=%s) …",
            model_id, quant, quant_method
        )

        # Build vLLM initialization kwargs
        llm_kwargs = {
            "model": model_id,
            "dtype": "bfloat16",
            "gpu_memory_utilization": 0.90,
            # Default 4K caps KV cache to fit comfortably on an RTX 6000; override
            # via ModelSpec.max_seq_length in config for models needing more context.
            "max_model_len": config.get("max_seq_length") or 4096,
            "enforce_eager": True,
            "load_format": "safetensors",
            "compilation_config": {"backend": "none"},  # Disable torch.compile (Gemma-4 incompatibility)
        }

        # Apply quantization only if explicitly requested
        if quant_method:
            llm_kwargs["quantization"] = quant_method

        # Enable KV cache if configured
        if config.get("kv_cache", False):
            llm_kwargs["enable_prefix_caching"] = config.get("prefix_cache", False)

        # Advanced vLLM options (Python API only)
        # Note: vLLM serve CLI args (tool_call_parser, reasoning_parser, etc.)
        # are NOT supported here. Use them with 'vllm serve' command instead.

        if config.get("trust_remote_code"):
            llm_kwargs["trust_remote_code"] = True

        if config.get("max_num_seqs"):
            llm_kwargs["max_num_seqs"] = config["max_num_seqs"]

        if config.get("attention_backend"):
            logger.info("Note: attention_backend='%s' is for vLLM serve CLI, not Python API",
                       config["attention_backend"])

        self._llm = LLM(**llm_kwargs)
        self._model_id = model_id

    def generate(self, prompt: str, **kwargs) -> tuple[str, dict]:
        from vllm import SamplingParams

        sp = SamplingParams(
            max_tokens=kwargs.get("max_new_tokens", 256),
            temperature=kwargs.get("temperature", 0.0),
            top_p=kwargs.get("top_p", 1.0),
            repetition_penalty=kwargs.get("repetition_penalty", 1.0),
        )

        # Instruction-tuned models require turn-delimiter tokens from the chat
        # template; raw-text generate() causes repetition-loop degeneration.
        t0 = time.perf_counter()
        outputs = self._llm.chat([[{"role": "user", "content": prompt}]], sp)
        elapsed = time.perf_counter() - t0

        out = outputs[0].outputs[0]
        n_tokens = len(out.token_ids)

        # Extract response text (skip thinking/reasoning blocks if present)
        text = out.text.strip()

        # For models with reasoning (e.g., DiffusionGemma): extract just the response
        if "<answer>" in text or "<response>" in text:
            # Extract content between answer/response tags
            import re
            match = re.search(r"<(?:answer|response)>(.*?)</(?:answer|response)>", text, re.DOTALL)
            if match:
                text = match.group(1).strip()

        meta = {
            "ttft_s": getattr(out, "time_to_first_token", None),
            "tps": n_tokens / elapsed if elapsed > 0 else 0.0,
            "total_s": elapsed,
            "n_tokens": n_tokens,
        }
        return text, meta

    def cleanup(self) -> None:
        try:
            import torch
            del self._llm
            self._llm = None
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            logger.warning("vLLM cleanup error (GPU memory may not be freed): %s", e)


# ---------------------------------------------------------------------------
# OpenAI-API-compatible backend (OpenAI, OpenRouter, Gemini, self-hosted, …)
# ---------------------------------------------------------------------------

class OpenAICompatibleEngine(BaseInferenceEngine):
    """
    Generic client for any provider implementing the OpenAI /chat/completions
    schema: OpenAI itself, OpenRouter (proxies Claude, Gemini, Llama, etc.),
    Gemini's native OpenAI-compat endpoint, or a self-hosted server.

    No local GPU/VRAM telemetry applies here — hardware_monitor will simply
    report psutil-only CPU/RAM stats for these cells, which is expected.
    """

    def __init__(self) -> None:
        self._client = None
        self._model_name: Optional[str] = None

    def load_model(self, model_id: str, config: dict) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError(
                "The 'openai' package is not installed. Run: pip install openai"
            ) from e

        # Round-trip through APISpec so provider presets (base_url, api_key_env)
        # resolve the same way regardless of caller — config_parser.APISpec is
        # the single source of truth for provider presets, not duplicated here.
        from .config_parser import APISpec
        api_spec = APISpec(**(config.get("api_config") or {}))

        key_env = api_spec.resolved_key_env()
        api_key = os.environ.get(key_env)
        if not api_key:
            raise RuntimeError(
                f"Environment variable '{key_env}' is not set. "
                f"Export it before running (e.g. export {key_env}=sk-...)."
            )

        base_url = api_spec.resolved_base_url()
        self._model_name = api_spec.model or model_id
        logger.info(
            "Configuring OpenAI-compatible client: provider=%s base_url=%s model=%s",
            api_spec.provider, base_url, self._model_name,
        )

        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=api_spec.timeout_s,
            max_retries=api_spec.max_retries,
            default_headers=api_spec.extra_headers or None,
        )

    def generate(self, prompt: str, **kwargs) -> tuple[str, dict]:
        t0 = time.perf_counter()
        resp = self._client.chat.completions.create(
            model=self._model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=kwargs.get("max_new_tokens", 256),
            temperature=kwargs.get("temperature", 0.0),
            top_p=kwargs.get("top_p", 1.0),
        )
        elapsed = time.perf_counter() - t0

        text = (resp.choices[0].message.content or "").strip()
        usage = getattr(resp, "usage", None)
        n_tokens = getattr(usage, "completion_tokens", None) or max(len(text.split()), 1)

        meta = {
            "ttft_s": None,  # not exposed by non-streaming chat completions
            "tps": n_tokens / elapsed if elapsed > 0 else 0.0,
            "total_s": elapsed,
            "n_tokens": n_tokens,
        }
        return text, meta

    def cleanup(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception as e:
                logger.debug("OpenAI client close error (non-fatal): %s", e)
        self._client = None


# ---------------------------------------------------------------------------
# Mock backend (no network, no GPU)
# ---------------------------------------------------------------------------

class MockInferenceEngine(BaseInferenceEngine):
    """
    Deterministic mock for testing the pipeline without GPU hardware or API calls.
    Simulates any model via simple linear latency.
    """

    _MOCK_RESPONSES = [
        "A", "B", "C", "D",
        "Paris", "42", "True", "False",
        "The answer is C.", "Option A is correct.",
    ]

    def __init__(self, base_latency_ms: int = 120, tps: int = 80, oom_prob: float = 0.0):
        self._base_latency = base_latency_ms / 1000.0
        self._tps = tps
        self._oom_prob = oom_prob
        self._model_id: Optional[str] = None

    def load_model(self, model_id: str, config: dict) -> None:
        self._model_id = model_id
        logger.info("[MOCK] Loaded engine for model '%s'", model_id)

    def generate(self, prompt: str, **kwargs) -> tuple[str, dict]:
        if self._oom_prob > 0 and random.random() < self._oom_prob:
            raise RuntimeError("CUDA out of memory (simulated OOM for testing)")

        max_new_tokens = kwargs.get("max_new_tokens", 256)
        simulated_tokens = min(max_new_tokens, random.randint(8, 64))
        total_s = self._base_latency + simulated_tokens / self._tps
        ttft_s = self._base_latency

        time.sleep(total_s)

        text = random.choice(self._MOCK_RESPONSES)
        meta = {
            "ttft_s": ttft_s,
            "tps": simulated_tokens / total_s,
            "total_s": total_s,
            "n_tokens": simulated_tokens,
        }
        return text, meta

    def cleanup(self) -> None:
        logger.debug("[MOCK] Cleanup called for '%s'", self._model_id)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_engine(backend: str, mock_cfg: Any | None = None) -> BaseInferenceEngine:
    if backend == "mock":
        kwargs: dict[str, Any] = {}
        if mock_cfg:
            kwargs = {
                "base_latency_ms": mock_cfg.base_latency_ms,
                "tps": mock_cfg.tokens_per_second,
                "oom_prob": mock_cfg.simulate_oom_probability,
            }
        return MockInferenceEngine(**kwargs)
    elif backend == "vllm":
        return VLLMInferenceEngine()
    elif backend == "api":
        return OpenAICompatibleEngine()
    else:
        return HFInferenceEngine()
