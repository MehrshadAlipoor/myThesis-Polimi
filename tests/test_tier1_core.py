"""Unit tests for the RES (Reasoning Efficiency Score) computation.

Uses the deterministic mock LLM client so the tests run offline.
"""

import sys
from pathlib import Path

import pytest

from nsclc_eval import config
from nsclc_eval.models import EfficiencyCategory, StepClassification
from nsclc_eval.testing import MockLLMClient
from nsclc_eval.tier1_core import (
    _extract_step_texts,
    calculate_res,
    classify_reasoning_steps,
    decompose_reasoning,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


@pytest.fixture(autouse=True)
def offline_client():
    config.configure(client_instance=MockLLMClient(), judge_models=[
                     "mock-judge"], seed=42, temperature=0.2)


def _make_assessments(categories):
    return [
        StepClassification(
            step_id=i,
            classification=cat,
            rationale="test",
        )
        for i, cat in enumerate(categories, 1)
    ]


def test_calculate_res_all_reasoning():
    assessments = _make_assessments([EfficiencyCategory.REASONING] * 4)
    res, reasoning, citation, total, stats = calculate_res(assessments)
    assert res == 100.0
    assert reasoning == 4
    assert citation == 0
    assert total == 4
    assert stats["Reasoning"]["count"] == 4


def test_calculate_res_mixed():
    assessments = _make_assessments(
        [
            EfficiencyCategory.REASONING,
            EfficiencyCategory.REASONING,
            EfficiencyCategory.CITATION,
            EfficiencyCategory.REDUNDANCY,
        ]
    )
    res, reasoning, citation, total, stats = calculate_res(assessments)
    assert res == 50.0
    assert reasoning == 2
    assert citation == 1
    assert total == 4


def test_calculate_res_empty():
    res, reasoning, citation, total, stats = calculate_res([])
    assert res == 0
    assert total == 0


def test_extract_step_texts_explicit_steps():
    blob = "<Step 1> First step here.\n<Step 2> Second step here."
    steps = _extract_step_texts(blob)
    assert steps == ["First step here.", "Second step here."]


def test_extract_step_texts_empty():
    assert _extract_step_texts("") == []


def test_decompose_and_classify_with_mock():
    raw = "Some raw agent reasoning text for the mock."
    atomic = decompose_reasoning(raw)
    steps = _extract_step_texts(atomic)
    assert len(steps) == 3  # deterministic mock returns 3 steps
    assessments = classify_reasoning_steps(atomic)
    assert len(assessments) == 3
    assert all(a.classification == EfficiencyCategory.REASONING for a in assessments)


def test_classify_pads_missing_steps():
    # Mock returns one entry per input <Step N>, so all steps are covered.
    blob = "<Step 1> a\n<Step 2> b\n<Step 3> c\n<Step 4> d"
    assessments = classify_reasoning_steps(blob, steps=["a", "b", "c", "d"])
    assert [a.step_id for a in assessments] == [1, 2, 3, 4]
