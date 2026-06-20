"""Prompt templates with key-fallback schema tolerance."""

from __future__ import annotations

SYSTEM_PROMPT = (
    """
    You are a law graduate taking the second phase of the Brazilian Bar Association
    (OAB) exam, organized by FGV. Your task is to answer the essay questions and
    prepare a legal document, demonstrating your legal knowledge, reasoning ability, and
    skill in applying relevant legislation and jurisprudence to the presented case.
    ATTENTION
    When preparing the texts for the practical-professional document and answers to the
    essay questions, you must include all necessary data without producing any identification or
    information beyond what is provided and permitted in the statements contained in the exam booklet.
    The omission of data that is legally required or necessary for the correct solution of the proposed
    problem will result in point deductions. You must be careful not to generate any different
    data that could create an identifying mark.

    The detection of any identifying mark in the space designated for the transcription of
    the final texts will result in the annulment of the practical-professional exam and your
    elimination. For example, when closing the document, you should opt to use only
    "ellipsis" or "XXX", that is: date "..." or Date "XXX", location "..." or Location "XXX",
    Attorney "..." or Attorney "XXX", OAB registration "..." or OAB Registration "XXX".
    Note that in the body of your answers, you should not create any data that generates
    an identification mark.
    OBSERVATIONS
    PRACTICAL-PROFESSIONAL DOCUMENT: The document must cover all legal
    grounds that can be used to support the claim. Simply mentioning or transcribing
    the legal provision does not earn points.
    QUESTION: You must provide reasoning for your answers. Merely citing the legal
    provision does not earn points.
    From now on, all your answers will compose the final text (not the draft booklet)
    """
)

MULTIPLE_CHOICE_SYSTEM_PROMPT = (
    """
    You are an expert educator answering multiple-choice questions from the Brazilian ENEM (Exame Nacional do Ensino Médio) exam.

    CRITICAL: Your response MUST be ONLY the letter of the correct answer (A, B, C, D, or E).
    - Do NOT include the full option text
    - Do NOT include any explanation
    - Do NOT add any other text or punctuation
    - Respond with EXACTLY ONE letter: A, B, C, D, or E
    """
)

_QUESTION_KEYS = ["question", "text", "input", "prompt"]
_ANSWER_KEYS   = ["answer", "label", "output", "target"]
_CHOICES_KEYS  = ["alternatives", "choices"]

# Marker emitted into multiple-choice prompts below; orchestrator and evaluator
# both check for it to detect MC questions, so it's defined once here.
OPTIONS_MARKER = "Options:"


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
        return "\n".join(f"  {label}. {t}" for label, t in zip(labels, texts))
    # plain list
    return "\n".join(f"  {chr(65 + i)}. {c}" for i, c in enumerate(choices_raw))


def build_prompt(record: dict) -> str:
    """Build a zero-shot prompt tolerating varied HF dataset schemas."""
    question = _extract(record, _QUESTION_KEYS)
    if question is None:
        raise KeyError(
            f"Record has none of the expected question keys {_QUESTION_KEYS}: {list(record.keys())}"
        )

    # Clean anonymization markers from datasets like ENEM
    question = question.replace("[[placeholder]]", "").strip()

    for ck in _CHOICES_KEYS:
        if ck in record and record[ck]:
            options = _format_options(record[ck])
            return f"{MULTIPLE_CHOICE_SYSTEM_PROMPT}\n\nQuestion: {question}\n\n{OPTIONS_MARKER}\n{options}\n\nAnswer:"

    return f"{SYSTEM_PROMPT}\n\nQuestion: {question}\n\nAnswer:"


def extract_reference_answer(record: dict) -> str:
    """Extract the ground-truth label/answer from a record."""
    for k in _ANSWER_KEYS:
        if k in record:
            return str(record[k]).strip()
    return ""
