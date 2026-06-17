"""Inference engine abstraction: HF Transformers, vLLM, and Mock backends."""

from __future__ import annotations

import gc
import logging
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
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

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
            logger.debug("HF cleanup error: %s", e)


# ---------------------------------------------------------------------------
# vLLM backend
# ---------------------------------------------------------------------------

class VLLMInferenceEngine(BaseInferenceEngine):

    def __init__(self) -> None:
        self._llm = None
        self._sampling_params = None

    def load_model(self, model_id: str, config: dict) -> None:
        try:
            from vllm import LLM, SamplingParams
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
            "max_model_len": 4096,  # Cap at 4K tokens to fit KV cache on RTX 6000
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
        
        self._llm = LLM(**llm_kwargs)

    def generate(self, prompt: str, **kwargs) -> tuple[str, dict]:
        from vllm import SamplingParams

        sp = SamplingParams(
            max_tokens=kwargs.get("max_new_tokens", 256),
            temperature=kwargs.get("temperature", 0.0),
            top_p=kwargs.get("top_p", 1.0),
            repetition_penalty=kwargs.get("repetition_penalty", 1.0),
        )

        t0 = time.perf_counter()
        outputs = self._llm.generate([prompt], sp)
        elapsed = time.perf_counter() - t0

        out = outputs[0].outputs[0]
        n_tokens = len(out.token_ids)
        meta = {
            "ttft_s": getattr(out, "time_to_first_token", None),
            "tps": n_tokens / elapsed if elapsed > 0 else 0.0,
            "total_s": elapsed,
            "n_tokens": n_tokens,
        }
        return out.text.strip(), meta

    def cleanup(self) -> None:
        try:
            import torch
            del self._llm
            self._llm = None
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            logger.debug("vLLM cleanup error: %s", e)


# ---------------------------------------------------------------------------
# Mock backend (Sabiá-4 stub)
# ---------------------------------------------------------------------------

class MockInferenceEngine(BaseInferenceEngine):
    """
    Deterministic mock for testing the pipeline without GPU hardware.
    Simulates Maritaca Sabiá-4 or any model via simple linear delay.
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
    else:
        return HFInferenceEngine()
