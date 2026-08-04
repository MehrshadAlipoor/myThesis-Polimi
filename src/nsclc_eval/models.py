"""Pydantic data models for the NSCLC therapy-agent evaluation framework.

All models use ``model_validator(mode='before')`` / ``field_validator`` to
tolerate the alternative field names that small open-weight LLMs frequently
return, so the structured ``response_format`` parsing is robust against
hallucinated JSON schemas.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# =====================================================================
# TIER 1 CORE — Reasoning Efficiency (RES)
# =====================================================================

class EfficiencyCategory(str, Enum):
    CITATION = "Citation"
    REPETITION = "Repetition"
    REASONING = "Reasoning"
    REDUNDANCY = "Redundancy"


class StepClassification(BaseModel):
    step_id: int
    original_text: str = Field(description="The original text/log from the agent for this step")
    claim: str
    classification: EfficiencyCategory
    rationale: str = Field(description="Explanation for why this step falls into this category")

    @field_validator('classification', mode='before')
    @classmethod
    def parse_hallucinated_class(cls, v):
        def _map_to_enum(val_str):
            s = str(val_str).title()
            for valid_cat in ["Citation", "Repetition", "Reasoning", "Redundancy"]:
                if valid_cat in s:
                    return valid_cat
            return "Reasoning"
        if isinstance(v, dict):
            for key in ['classification', 'type', 'category', 'value', 'EfficiencyCategory']:
                if key in v:
                    return _map_to_enum(v[key])
            for k, val in v.items():
                if val == 1 or val is True:
                    return _map_to_enum(k)
        return _map_to_enum(v)


# =====================================================================
# TIER 2 — Clinical Effectiveness
# =====================================================================

class TreatmentCategory(str, Enum):
    IOCT = "Immunotherapy + Chemotherapy (IOCT)"
    IO = "Immunotherapy alone (IO)"


class AccuracyExtraction(BaseModel):
    raw_prediction: str = Field(description="Raw text extracted from the model's final decision")
    predicted_label: int = Field(description="Binary label: 0 (IO) or 1 (IOCT)")
    category: TreatmentCategory = Field(description="Categorical label for the predicted treatment")
    rationale: str = Field(description="Explanation for the classification")

    @field_validator('category', mode='before')
    @classmethod
    def parse_hallucinated_category(cls, v):
        def _map_to_enum(val_str):
            s = str(val_str).upper()
            if "IOCT" in s or "CHEMO" in s:
                return "Immunotherapy + Chemotherapy (IOCT)"
            return "Immunotherapy alone (IO)"
        if isinstance(v, dict):
            for key in ['category', 'type', 'value', 'TreatmentCategory']:
                if key in v:
                    return _map_to_enum(v[key])
            for k, val in v.items():
                if val == 1 or val is True:
                    return _map_to_enum(k)
        return _map_to_enum(v)


# =====================================================================
# COMPLETENESS METRICS
# =====================================================================

class MentionedItem(BaseModel):
    checklist_item: str
    mentioned: bool
    extracted_value: Optional[str] = ""
    rationale: Optional[str] = ""
    step_numbers: List[int] = []

    @model_validator(mode='before')
    @classmethod
    def remap_llm_fields(cls, data):
        if isinstance(data, dict):
            if 'checklist_item' not in data:
                for alt in ('item', 'name', 'factor'):
                    if alt in data:
                        data['checklist_item'] = data.pop(alt)
                        break
            if 'mentioned' not in data:
                for alt in ('is_mentioned', 'present', 'found'):
                    if alt in data:
                        data['mentioned'] = data.pop(alt)
                        break
            if 'extracted_value' not in data:
                for alt in ('value', 'actual_value'):
                    if alt in data:
                        data['extracted_value'] = data.pop(alt)
                        break
            if data.get('extracted_value') is not None:
                data['extracted_value'] = str(data['extracted_value'])
        return data


class ReasoningCompletenessExtraction(BaseModel):
    items: List[MentionedItem]


# =====================================================================
# TIER 1 EXTENDED — Factuality
# The LLM often returns {claim, status} or {claim, classification}
# instead of {step_number, claim, referenced_field, actual_value, verdict}.
# model_validator remaps these alternative names.
# =====================================================================

class FactualClaim(BaseModel):
    step_number: int = 0
    claim: str = ""
    referenced_field: str = ""
    actual_value: Any = ""
    verdict: str = "unsupported"

    @model_validator(mode='before')
    @classmethod
    def remap_llm_fields(cls, data):
        if isinstance(data, dict):
            # verdict aliases
            if 'verdict' not in data:
                for alt in ('status', 'classification', 'result', 'check'):
                    if alt in data:
                        data['verdict'] = data.pop(alt)
                        break
            # step_number aliases
            if 'step_number' not in data:
                for alt in ('step', 'step_id', 'step_num', 'number'):
                    if alt in data:
                        data['step_number'] = data.pop(alt)
                        break
            # referenced_field aliases
            if 'referenced_field' not in data:
                for alt in ('field', 'reference', 'ref_field', 'data_field', 'source_field', 'patient_field'):
                    if alt in data:
                        data['referenced_field'] = data.pop(alt)
                        break
            # actual_value aliases
            if 'actual_value' not in data:
                for alt in ('actual', 'value', 'ground_truth', 'expected', 'patient_value', 'true_value'):
                    if alt in data:
                        data['actual_value'] = data.pop(alt)
                        break
        return data


class FactualityResult(BaseModel):
    claims: List[FactualClaim] = []
    total_claims: int = 0
    correct_claims: int = 0
    incorrect_claims: int = 0
    unsupported_claims: int = 0
    factuality_score: float = 0.0

    @staticmethod
    def _claim_value(claim: Any, *keys: str) -> Any:
        """Read a field from either a dict or a pydantic model instance."""
        for key in keys:
            if isinstance(claim, dict):
                val = claim.get(key)
            else:
                val = getattr(claim, key, None)
            if val:
                return val
        return None

    @model_validator(mode='before')
    @classmethod
    def remap_and_recount(cls, data):
        """If LLM omits count fields, recompute from claims list."""
        if isinstance(data, dict):
            claims = data.get('claims', [])
            if claims and not data.get('total_claims'):
                data['total_claims'] = len(claims)
            # Recount from actual claim verdicts if counts seem wrong
            if claims and data.get('total_claims', 0) > 0:
                verdicts = []
                for c in claims:
                    v = (
                        cls._claim_value(c, 'verdict', 'status', 'classification')
                        or 'unsupported'
                    )
                    verdicts.append(str(v).lower())
                data['correct_claims'] = sum(1 for v in verdicts if v == 'correct')
                data['incorrect_claims'] = sum(1 for v in verdicts if v == 'incorrect')
                data['unsupported_claims'] = sum(1 for v in verdicts if v not in ('correct', 'incorrect'))
                total = data['total_claims']
                data['factuality_score'] = (data['correct_claims'] / total * 100) if total else 0.0
        return data


# =====================================================================
# TIER 1 EXTENDED — Faithfulness
# The LLM often returns {step, gap} instead of {step_number, description}.
# =====================================================================

class Contradiction(BaseModel):
    step_a: int = 0
    step_b: int = 0
    description: str = ""

    @model_validator(mode='before')
    @classmethod
    def remap_llm_fields(cls, data):
        if isinstance(data, dict):
            if 'step_a' not in data:
                for alt in ('step1', 'first_step', 'stepA'):
                    if alt in data:
                        data['step_a'] = data.pop(alt)
                        break
            if 'step_b' not in data:
                for alt in ('step2', 'second_step', 'stepB'):
                    if alt in data:
                        data['step_b'] = data.pop(alt)
                        break
            if 'description' not in data:
                for alt in ('desc', 'explanation', 'detail', 'reason', 'text'):
                    if alt in data:
                        data['description'] = data.pop(alt)
                        break
        return data


class LogicalGap(BaseModel):
    step_number: int = 0
    description: str = ""

    @model_validator(mode='before')
    @classmethod
    def remap_llm_fields(cls, data):
        if isinstance(data, dict):
            # step_number aliases
            if 'step_number' not in data:
                for alt in ('step', 'step_id', 'step_num', 'number'):
                    if alt in data:
                        data['step_number'] = data.pop(alt)
                        break
            # description aliases
            if 'description' not in data:
                for alt in ('gap', 'desc', 'explanation', 'detail', 'reason', 'text'):
                    if alt in data:
                        data['description'] = data.pop(alt)
                        break
        return data


class FaithfulnessResult(BaseModel):
    contradictions: List[Contradiction] = []
    conclusion_follows: bool = True
    logical_gaps: List[LogicalGap] = []
    faithfulness_score: int = 3
    summary: str = ""

    @model_validator(mode='before')
    @classmethod
    def remap_llm_fields(cls, data):
        if isinstance(data, dict):
            if 'conclusion_follows' not in data:
                for alt in ('conclusion', 'follows', 'conclusion_supported', 'supported'):
                    if alt in data:
                        data['conclusion_follows'] = data.pop(alt)
                        break
            if 'faithfulness_score' not in data:
                for alt in ('score', 'rating', 'likert', 'likert_score'):
                    if alt in data:
                        data['faithfulness_score'] = data.pop(alt)
                        break
            if 'summary' not in data:
                for alt in ('explanation', 'rationale', 'assessment', 'overall'):
                    if alt in data:
                        data['summary'] = data.pop(alt)
                        break
        return data


# =====================================================================
# TIER 1 EXTENDED — Guideline Adherence (GAR)
# =====================================================================

class GuidelineAdherenceStep(BaseModel):
    step_number: int
    adheres_to_guideline: bool = Field(
        description="True if the step aligns with or does not contradict the guidelines. False if it violates the guidelines."
    )
    rationale: str = Field(
        description="Explanation of why this step adheres or fails to adhere to the guidelines provided."
    )


class GuidelineAdherenceResult(BaseModel):
    steps: List[GuidelineAdherenceStep]


# =====================================================================
# INPUT DATA — Patient Record
# =====================================================================

@dataclass
class PatientRecord:
    patient_id: str
    i3lung_id: str
    orchestration_model_name: str
    raw_reasoning: str
    final_decision: str
