"""Scoring logic — exact/normalized match and choice extraction directly
determine reported accuracy numbers, so regressions here are easy to miss
visually but make every downstream benchmark result wrong."""

import pytest

from src.evaluator import Evaluator, HeuristicJudgeMock, _extract_choice, _normalize


def test_normalize_strips_punctuation_and_articles():
    # "the" and the trailing "a" are both articles and get stripped too.
    assert _normalize("The Answer Is: A.") == "answer is"


def test_extract_choice_finds_standalone_letter():
    assert _extract_choice("A. mudança de comportamento.") == "A"
    assert _extract_choice("  c  ") == "C"


def test_extract_choice_falls_back_to_first_char_when_no_word_boundary_letter():
    assert _extract_choice("42") == "4"


def test_extract_choice_empty_string_returns_empty():
    assert _extract_choice("") == ""


class TestEvaluateBatch:
    def test_exact_match_on_plain_text(self):
        ev = Evaluator()
        summary = ev.evaluate_batch(["Q: capital of France?"], ["Paris"], ["Paris"])
        assert summary.exact_match_pct == 100.0

    def test_multiple_choice_extracts_letter_from_verbose_prediction(self):
        ev = Evaluator()
        prompt = "Question: ...\n\nOptions:\n  A. foo\n  B. bar\n\nAnswer:"
        summary = ev.evaluate_batch([prompt], ["A"], ["A. foo, the correct choice"])
        assert summary.exact_match_pct == 100.0

    def test_multiple_choice_wrong_letter_is_not_a_match(self):
        ev = Evaluator()
        prompt = "Question: ...\n\nOptions:\n  A. foo\n  B. bar\n\nAnswer:"
        summary = ev.evaluate_batch([prompt], ["A"], ["B. bar"])
        assert summary.exact_match_pct == 0.0

    def test_mismatched_lengths_raise_instead_of_silently_truncating(self):
        ev = Evaluator()
        with pytest.raises(ValueError):
            ev.evaluate_batch(["p1", "p2"], ["r1"], ["pred1"])

    def test_judge_none_by_default_leaves_judge_score_zero(self):
        ev = Evaluator()
        summary = ev.evaluate_batch(["q"], ["A"], ["B"])
        assert summary.judge_avg_score == 0.0

    def test_explicit_judge_is_used_and_scored(self):
        ev = Evaluator(judge=HeuristicJudgeMock())
        summary = ev.evaluate_batch(["q"], ["A"], ["A"])
        assert summary.judge_avg_score == 1.0

    def test_legacy_use_judge_flag_still_works(self):
        ev = Evaluator(use_judge=True)
        summary = ev.evaluate_batch(["q"], ["A"], ["A"])
        assert summary.judge_avg_score == 1.0


class TestHeuristicJudgeMock:
    def test_exact_text_match_scores_correct(self):
        judge = HeuristicJudgeMock()
        score, label = judge.judge("q", "Paris", "Paris")
        assert score == 1.0
        assert label == "correct"

    def test_matching_choice_letter_scores_correct(self):
        judge = HeuristicJudgeMock()
        score, label = judge.judge("q", "A", "A. some long option text")
        assert score == 1.0
        assert label == "correct"

    def test_unrelated_answer_scores_incorrect(self):
        judge = HeuristicJudgeMock()
        score, label = judge.judge("q", "Paris", "Tokyo")
        assert score == 0.0
        assert label == "incorrect"
