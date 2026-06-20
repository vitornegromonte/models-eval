"""Accuracy evaluator with Exact Match, normalized match, and LLM-as-a-judge (mock or real API)."""

from __future__ import annotations

import json
import logging
import re
import string
from dataclasses import dataclass, field
from typing import Optional, Protocol

from prompts.templates import OPTIONS_MARKER

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
# Judge protocol — anything with .judge(question, reference, prediction)
# ---------------------------------------------------------------------------

class Judge(Protocol):
    def judge(self, question: str, reference: str, prediction: str) -> tuple[float, str]: ...


# ---------------------------------------------------------------------------
# Heuristic judge (no network calls — string-similarity mock)
# ---------------------------------------------------------------------------

class HeuristicJudgeMock:
    """
    Deterministic, offline judge based on normalized-string/choice matching.
    Useful for mock-backend sweeps where no API call should be made at all.
    For a real LLM-as-a-judge, use APIJudge below (works with Maritaca's
    Sabiá models too — see the 'maritaca' provider preset in config_parser.APISpec).
    """

    def judge(self, question: str, reference: str, prediction: str) -> tuple[float, str]:
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


# ---------------------------------------------------------------------------
# Real LLM judge — a judge is just an OpenAI-API-compatible model with a
# scoring prompt, so it reuses OpenAICompatibleEngine instead of duplicating
# client setup / API-calling code. Works with OpenAI, OpenRouter (proxies
# Claude/Gemini/etc.), Maritaca's Sabiá models, or a self-hosted server —
# whichever provider is named in the APISpec config (config_parser.APISpec).
# ---------------------------------------------------------------------------

class APIJudge:
    _SYSTEM = (
        "You are an impartial judge. Given a question, a reference answer, and a "
        "model prediction, score the prediction on a scale from 0.0 (completely wrong) "
        "to 1.0 (perfectly correct). Reply with ONLY a JSON object, no other text: "
        '{"score": <float 0.0-1.0>, "label": "<correct|partial|incorrect>"}.'
    )

    def __init__(self, api_config: dict):
        from .inference_engine import OpenAICompatibleEngine

        model_name = api_config.get("model") or "gpt-4o-mini"
        self._engine = OpenAICompatibleEngine()
        self._engine.load_model(model_name, {"api_config": api_config})

    def judge(self, question: str, reference: str, prediction: str) -> tuple[float, str]:
        prompt = (
            f"{self._SYSTEM}\n\n"
            f"Question:\n{question}\n\n"
            f"Reference answer:\n{reference}\n\n"
            f"Model prediction:\n{prediction}\n\n"
            "Score the prediction now."
        )
        try:
            text, _meta = self._engine.generate(prompt, max_new_tokens=100, temperature=0.0)
            match = re.search(r"\{.*\}", text, re.DOTALL)
            payload = json.loads(match.group(0) if match else text)
            score = float(payload.get("score", 0.0))
            label = str(payload.get("label", "incorrect"))
            return max(0.0, min(1.0, score)), label
        except Exception as e:
            logger.warning("APIJudge call failed, scoring as incorrect: %s", e)
            return 0.0, "incorrect"


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    def __init__(self, use_judge: bool = False, judge: Optional["Judge"] = None):
        """
        Pass `judge=` with a Judge instance (HeuristicJudgeMock or APIJudge) for
        explicit control. The legacy `use_judge=True` with no `judge=` still
        defaults to the deterministic mock for backward compatibility.
        """
        if judge is not None:
            self._judge = judge
        elif use_judge:
            self._judge = HeuristicJudgeMock()
        else:
            self._judge = None
        self._use_judge = self._judge is not None

    def evaluate_batch(
        self,
        prompts: list[str],
        references: list[str],
        predictions: list[str],
    ) -> EvalSummary:
        if not len(prompts) == len(references) == len(predictions):
            raise ValueError(
                f"prompts/references/predictions length mismatch: "
                f"{len(prompts)}/{len(references)}/{len(predictions)}"
            )

        records: list[EvalRecord] = []
        judge_scores: list[float] = []

        for prompt, ref, pred in zip(prompts, references, predictions):
            # For multiple-choice questions: aggressively extract just the first choice letter
            processed_pred = pred.strip()
            if OPTIONS_MARKER in prompt:
                # Try to find any A-E letter in the prediction
                m = re.search(r"[A-Ea-e]", pred)
                if m:
                    processed_pred = m.group(0).upper()
                else:
                    # Fallback: use standard extraction
                    extracted = _extract_choice(pred)
                    if extracted and len(extracted) == 1 and extracted.isalpha():
                        processed_pred = extracted

            em  = ref.strip() == processed_pred
            nem = _normalize(ref) == _normalize(processed_pred)

            # Also try single-char choice comparison
            if not nem:
                nem = _extract_choice(ref) == _extract_choice(processed_pred)

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
