"""Prompt templates with key-fallback schema tolerance."""

from __future__ import annotations

SYSTEM_PROMPT = (
    "You are a concise and accurate assistant. "
    "Answer the following question with only the correct answer. "
    "Do not add explanations."
)

_QUESTION_KEYS = ["question", "text", "input", "prompt"]
_ANSWER_KEYS   = ["answer", "label", "output", "target"]
_CHOICES_KEYS  = ["alternatives", "choices"]


def _extract(record: dict, keys: list[str]) -> str | None:
    for k in keys:
        if k in record:
            val = record[k]
            return str(val) if not isinstance(val, list) else "; ".join(str(v) for v in val)
    return None


def _format_options(choices_raw: dict | list) -> str:
    """Render multiple-choice options from dict (ENEM) or list (MMLU) format."""
    if isinstance(choices_raw, dict):
        # ENEM: {"A": "texto A", "B": "texto B", ...}
        if all(isinstance(v, str) for v in choices_raw.values()):
            return "\n".join(f"  {k}. {v}" for k, v in sorted(choices_raw.items()))
        # MMLU HF: {"text": [...], "label": [...]}
        texts  = choices_raw.get("text", [])
        labels = choices_raw.get("label", [chr(65 + i) for i in range(len(texts))])
        return "\n".join(f"  {l}. {t}" for l, t in zip(labels, texts))
    # plain list
    return "\n".join(f"  {chr(65 + i)}. {c}" for i, c in enumerate(choices_raw))


def build_prompt(record: dict) -> str:
    """Build a zero-shot prompt tolerating varied HF dataset schemas."""
    question = _extract(record, _QUESTION_KEYS)
    if question is None:
        raise KeyError(
            f"Record has none of the expected question keys {_QUESTION_KEYS}: {list(record.keys())}"
        )

    for ck in _CHOICES_KEYS:
        if ck in record and record[ck]:
            options = _format_options(record[ck])
            return f"{SYSTEM_PROMPT}\n\nQuestion: {question}\n\nOptions:\n{options}\n\nAnswer:"

    return f"{SYSTEM_PROMPT}\n\nQuestion: {question}\n\nAnswer:"


def extract_reference_answer(record: dict) -> str:
    """Extract the ground-truth label/answer from a record."""
    for k in _ANSWER_KEYS:
        if k in record:
            return str(record[k]).strip()
    return ""
