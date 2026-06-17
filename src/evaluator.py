"""Accuracy evaluator with Exact Match and Sabiá-4 (Maritaca) LLM-judge mock."""

from __future__ import annotations

import logging
import re
import string
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class EvalRecord:
    prompt: str
    reference: str
    prediction: str
    exact_match: bool = False
    normalized_match: bool = False
    judge_score: Optional[float] = None
    judge_label: Optional[str] = None


@dataclass
class EvalSummary:
    total: int = 0
    exact_matches: int = 0
    normalized_matches: int = 0
    judge_avg_score: float = 0.0
    exact_match_pct: float = 0.0
    normalized_match_pct: float = 0.0
    records: list[EvalRecord] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

_PUNCT = set(string.punctuation)
_ARTICLES = {"a", "an", "the"}


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = "".join(ch for ch in text if ch not in _PUNCT)
    tokens = [t for t in text.split() if t not in _ARTICLES]
    return " ".join(tokens)


def _extract_choice(text: str) -> str:
    """Pull a single letter choice (A-F) from free-form generation output."""
    m = re.search(r"\b([A-Fa-f])\b", text)
    if m:
        return m.group(1).upper()
    if text.strip():
        return text.strip()[0].upper()
    return text


# ---------------------------------------------------------------------------
# Sabiá-4 LLM judge (mock)
# ---------------------------------------------------------------------------

class Sabia4JudgeMock:
    """
    Deterministic mock of the Maritaca Sabiá-4 API for correctness judgement.
    In production, replace _call_api() with actual HTTP requests to the
    Maritaca endpoint (api.maritaca.ai/chat/completions).
    """

    _SYSTEM = (
        "You are an impartial judge. Given a question, a reference answer, and a "
        "model prediction, score the prediction on a scale from 0.0 (completely wrong) "
        "to 1.0 (perfectly correct). Reply ONLY with a JSON object: "
        '{"score": <float>, "label": "<correct|partial|incorrect>"}.'
    )

    def judge(self, question: str, reference: str, prediction: str) -> tuple[float, str]:
        """Return (score, label). Uses mock logic; swap for real API call."""
        norm_ref  = _normalize(reference)
        norm_pred = _normalize(prediction)

        if norm_ref == norm_pred:
            return 1.0, "correct"

        ref_choice  = _extract_choice(reference)
        pred_choice = _extract_choice(prediction)
        if ref_choice and ref_choice == pred_choice:
            return 1.0, "correct"

        if norm_ref and norm_pred and (norm_ref in norm_pred or norm_pred in norm_ref):
            return 0.5, "partial"

        return 0.0, "incorrect"

    # Stub for real Maritaca API integration:
    # def _call_api(self, payload: dict) -> dict:
    #     import httpx
    #     resp = httpx.post(
    #         "https://api.maritaca.ai/api/chat/completions",
    #         headers={"Authorization": f"Key {os.environ['MARITACA_API_KEY']}"},
    #         json=payload,
    #         timeout=30,
    #     )
    #     resp.raise_for_status()
    #     return resp.json()


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    def __init__(self, use_judge: bool = False):
        self._use_judge = use_judge
        self._judge = Sabia4JudgeMock() if use_judge else None

    def evaluate_batch(
        self,
        prompts: list[str],
        references: list[str],
        predictions: list[str],
    ) -> EvalSummary:
        assert len(prompts) == len(references) == len(predictions)

        records: list[EvalRecord] = []
        judge_scores: list[float] = []

        for prompt, ref, pred in zip(prompts, references, predictions):
            em  = ref.strip() == pred.strip()
            nem = _normalize(ref) == _normalize(pred)

            # Also try single-char choice comparison
            if not nem:
                nem = _extract_choice(ref) == _extract_choice(pred)

            rec = EvalRecord(
                prompt=prompt,
                reference=ref,
                prediction=pred,
                exact_match=em,
                normalized_match=nem,
            )

            if self._judge:
                score, label = self._judge.judge(prompt, ref, pred)
                rec.judge_score = score
                rec.judge_label = label
                judge_scores.append(score)

            records.append(rec)

        n = len(records)
        exact   = sum(r.exact_match      for r in records)
        normed  = sum(r.normalized_match for r in records)
        j_avg   = sum(judge_scores) / len(judge_scores) if judge_scores else 0.0

        summary = EvalSummary(
            total=n,
            exact_matches=exact,
            normalized_matches=normed,
            judge_avg_score=j_avg,
            exact_match_pct=exact / n * 100 if n else 0.0,
            normalized_match_pct=normed / n * 100 if n else 0.0,
            records=records,
        )
        return summary

    def evaluate_single(self, prompt: str, reference: str, prediction: str) -> EvalRecord:
        summary = self.evaluate_batch([prompt], [reference], [prediction])
        return summary.records[0]
