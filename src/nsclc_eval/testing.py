"""Offline mock LLM client for demos and tests.

Implements the subset of the OpenAI client interface used by the framework
(``chat.completions.create`` and ``beta.chat.completions.parse``) with
deterministic, plausible outputs, so the whole pipeline can run end-to-end
without a live ``llama-cpp-server``.

Use it with :func:`nsclc_eval.config.configure`:

.. code-block:: python

    from nsclc_eval import config
    from nsclc_eval.testing import MockLLMClient
    config.configure(client_instance=MockLLMClient(), judge_models=["mock-judge"])
"""

from __future__ import annotations

import re
from types import SimpleNamespace
from typing import Any, List

from .models import (
    AccuracyExtraction,
    EfficiencyCategory,
    FactualClaim,
    FactualityResult,
    GuidelineAdherenceResult,
    GuidelineAdherenceStep,
    MentionedItem,
    ReasoningCompletenessExtraction,
    StepClassification,
)

COMPLETENESS_CHECKLIST = [
    "histology",
    "pdl1",
    "ecog_ps",
    "comorbidities",
    "disease_burden",
    "io_biomarkers",
    "stage",
    "contraindications",
]


def _last_user_message(messages: List[dict]) -> str:
    """Return the text of the last ``user`` message in the list."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return str(msg.get("content", ""))
    return ""


def _has_chemo(text: str) -> bool:
    """Heuristic: does the decision text mention cytotoxic chemotherapy?"""
    return bool(
        re.search(
            r"carboplatin|cisplatin|paclitaxel|pemetrexed|chemotherapy|docetaxel|chemo",
            text,
            re.IGNORECASE,
        )
    )


class MockLLMClient:
    """A deterministic fake of the OpenAI client used by the framework."""

    # ------------------------------------------------------------------
    # Nested objects mimicking openai.ChatCompletion
    # ------------------------------------------------------------------
    class _Completions:
        def __init__(self, owner: "MockLLMClient", beta: bool):
            self._owner = owner
            self._beta = beta

        def create(self, model: str, messages: List[dict], **kwargs):
            content = self._owner._generate_content(messages)
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(content=content))]
            )

        def parse(self, model: str, messages: List[dict], response_format: Any, **kwargs):
            parsed = self._owner._generate_parsed(messages, response_format)
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(parsed=parsed))]
            )

    class _Beta:
        def __init__(self, owner: "MockLLMClient"):
            self._owner = owner
            self.chat = SimpleNamespace(
                completions=MockLLMClient._Completions(owner, beta=True)
            )

    def __init__(self, *, seed: int = 0):
        self._seed = seed

    @property
    def chat(self) -> SimpleNamespace:
        return SimpleNamespace(completions=self._Completions(self, beta=False))

    @property
    def beta(self) -> "_MockLLMClient._Beta":
        return self._Beta(self)

    # ------------------------------------------------------------------
    # Response generators
    # ------------------------------------------------------------------
    def _generate_content(self, messages: List[dict]) -> str:
        """Produce atomic-step text for the reformat stage."""
        user = _last_user_message(messages)
        # Keep it deterministic and independent of the real content.
        return (
            "<Step 1> The patient presents with the clinical findings described in the record.\n"
            "<Step 2> These findings are weighed against first-line treatment guidelines.\n"
            "<Step 3> A treatment recommendation is derived from the available evidence."
        )

    def _generate_parsed(self, messages: List[dict], response_format: Any):
        """Instantiate the requested Pydantic model with plausible values."""
        user = _last_user_message(messages)

        if response_format is StepClassification:
            return StepClassification(
                step_id=1,
                original_text="The patient presents with clinical findings.",
                claim="The patient presents with clinical findings.",
                classification=EfficiencyCategory.REASONING,
                rationale="Derives a conclusion that advances the reasoning.",
            )

        if response_format is AccuracyExtraction:
            predicted = 1 if _has_chemo(user) else 0
            return AccuracyExtraction(
                raw_prediction=user,
                predicted_label=predicted,
                category="Immunotherapy + Chemotherapy (IOCT)"
                if predicted == 1
                else "Immunotherapy alone (IO)",
                rationale="Mock heuristic based on the presence of chemotherapy agents.",
            )

        if response_format is ReasoningCompletenessExtraction:
            return ReasoningCompletenessExtraction(
                items=[
                    MentionedItem(
                        checklist_item=item,
                        mentioned=True,
                        extracted_value="mentioned",
                        rationale="Deterministic mock.",
                        step_numbers=[1],
                    )
                    for item in COMPLETENESS_CHECKLIST
                ]
            )

        if response_format is FactualityResult:
            return FactualityResult(
                claims=[
                    FactualClaim(
                        step_number=1,
                        claim="The patient has the documented diagnosis.",
                        referenced_field="histology",
                        actual_value="Adenocarcinoma",
                        verdict="correct",
                    )
                ],
                total_claims=1,
                correct_claims=1,
                incorrect_claims=0,
                unsupported_claims=0,
                factuality_score=100.0,
            )

        if response_format is GuidelineAdherenceResult:
            return GuidelineAdherenceResult(
                steps=[
                    GuidelineAdherenceStep(
                        step_number=1,
                        adheres_to_guideline=True,
                        rationale="Deterministic mock: step aligns with guidelines.",
                    )
                ]
            )

        # FaithfulnessResult (or anything unhandled) -> default
        from .models import FaithfulnessResult

        return FaithfulnessResult(
            contradictions=[],
            conclusion_follows=True,
            logical_gaps=[],
            faithfulness_score=5,
            summary="Deterministic mock: fully consistent reasoning.",
        )
