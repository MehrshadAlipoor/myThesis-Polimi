"""Pydantic data models for the NSCLC therapy-agent evaluation framework.

All models use ``model_validator(mode='before')`` / ``field_validator`` to
tolerate the alternative field names that small open-weight LLMs frequently
return, so the structured ``response_format`` parsing is robust against
hallucinated JSON schemas.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


def _map_to_enum(val_str) -> str:
    s = str(val_str).title()
    for valid_cat in ["Citation", "Repetition", "Reasoning", "Redundancy"]:
        if valid_cat in s:
            return valid_cat
    return "Reasoning"


def _coerce_step_int(v) -> int:
    """Tolerates 'step_1', 'Step 3', '2.', etc. -> int (0 on failure)."""
    if isinstance(v, str):
        match = re.search(r"\d+", v)
        if match:
            return int(match.group())
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


# =====================================================================
# TIER 1 CORE — Reasoning Efficiency (RES)
# =====================================================================

class EfficiencyCategory(str, Enum):
    CITATION = "Citation"
    REPETITION = "Repetition"
    REASONING = "Reasoning"
    REDUNDANCY = "Redundancy"


class StepClassification(BaseModel):
    step_id: int = 0
    classification: EfficiencyCategory
    rationale: str = ""

    @field_validator("classification", mode="before")
    @classmethod
    def parse_hallucinated_class(cls, v):
        if isinstance(v, dict):
            for key in ["classification", "type", "category", "value", "EfficiencyCategory"]:
                if key in v:
                    return _map_to_enum(v[key])
            for k, val in v.items():
                if val == 1 or val is True:
                    return _map_to_enum(k)
        return _map_to_enum(v)

    @field_validator("step_id", mode="before")
    @classmethod
    def coerce_step_id(cls, v):
        if isinstance(v, dict):
            for alt in ("step", "step_number", "step_num", "id", "number"):
                if alt in v:
                    v = v[alt]
                    break
        try:
            return int(v)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return 0


class StepClassificationBatch(BaseModel):
    """Batched classification: one entry per reasoning step, in a single call."""

    assessments: List[StepClassification] = []

    @model_validator(mode="before")
    @classmethod
    def remap_llm_fields(cls, data):
        if isinstance(data, list):
            data = {"assessments": data}
        if isinstance(data, dict) and "assessments" not in data:
            for alt in ("results", "classifications", "steps", "items"):
                if alt in data:
                    data["assessments"] = data.pop(alt)
                    break
        return data


# =====================================================================
# TIER 2 — Clinical Effectiveness
# =====================================================================

class TreatmentCategory(str, Enum):
    IOCT = "Immunotherapy + Chemotherapy (IOCT)"
    IO = "Immunotherapy alone (IO)"


class AccuracyExtraction(BaseModel):
    raw_prediction: str = Field(default="", description="Raw text extracted from the model's final decision")
    predicted_label: int = Field(description="Binary label: 0 (IO) or 1 (IOCT)")
    category: TreatmentCategory = Field(description="Categorical label for the predicted treatment")
    rationale: str = Field(default="", description="Explanation for the classification")

    @field_validator("category", mode="before")
    @classmethod
    def parse_hallucinated_category(cls, v):
        def _map_cat(val_str):
            s = str(val_str).upper()
            if "IOCT" in s or "CHEMO" in s:
                return "Immunotherapy + Chemotherapy (IOCT)"
            return "Immunotherapy alone (IO)"

        if isinstance(v, dict):
            for key in ["category", "type", "value", "TreatmentCategory"]:
                if key in v:
                    return _map_cat(v[key])
            for k, val in v.items():
                if val == 1 or val is True:
                    return _map_cat(k)
        return _map_cat(v)

    @field_validator("predicted_label", mode="before")
    @classmethod
    def coerce_label(cls, v):
        s = str(v).strip()
        return 1 if ("1" in s and "0" not in s or "ioct" in s.lower()) else 0


# =====================================================================
# COMPLETENESS METRICS
# =====================================================================

class MentionedItem(BaseModel):
    checklist_item: str
    mentioned: bool
    extracted_value: Optional[str] = ""
    rationale: Optional[str] = ""
    step_numbers: List[int] = []

    @model_validator(mode="before")
    @classmethod
    def remap_llm_fields(cls, data):
        if isinstance(data, dict):
            if "checklist_item" not in data:
                for alt in ("item", "name", "factor"):
                    if alt in data:
                        data["checklist_item"] = data.pop(alt)
                        break
            if "mentioned" not in data:
                for alt in ("is_mentioned", "present", "found"):
                    if alt in data:
                        data["mentioned"] = data.pop(alt)
                        break
            if "extracted_value" not in data:
                for alt in ("value", "actual_value"):
                    if alt in data:
                        data["extracted_value"] = data.pop(alt)
                        break
            if data.get("extracted_value") is not None:
                data["extracted_value"] = str(data["extracted_value"])
        return data


class ReasoningCompletenessExtraction(BaseModel):
    items: List[MentionedItem]


# =====================================================================
# TIER 1 EXTENDED — Factuality
# =====================================================================

class FactualClaim(BaseModel):
    step_number: int = 0
    claim: str = ""
    referenced_field: str = ""
    actual_value: Any = ""
    verdict: str = "unsupported"

    @model_validator(mode="before")
    @classmethod
    def remap_llm_fields(cls, data):
        if isinstance(data, dict):
            if "verdict" not in data:
                for alt in ("status", "classification", "result", "check"):
                    if alt in data:
                        data["verdict"] = data.pop(alt)
                        break
            if "step_number" not in data:
                for alt in ("step", "step_id", "step_num", "number"):
                    if alt in data:
                        data["step_number"] = data.pop(alt)
                        break
            if "referenced_field" not in data:
                for alt in ("field", "reference", "ref_field", "data_field", "source_field", "patient_field"):
                    if alt in data:
                        data["referenced_field"] = data.pop(alt)
                        break
            if "actual_value" not in data:
                for alt in ("actual", "value", "ground_truth", "expected", "patient_value", "true_value"):
                    if alt in data:
                        data["actual_value"] = data.pop(alt)
                        break
        return data

    @field_validator("step_number", mode="before")
    @classmethod
    def coerce_step_number(cls, v):
        return _coerce_step_int(v)

    @field_validator("referenced_field", mode="before")
    @classmethod
    def coerce_referenced_field(cls, v):
        if v is None:
            return ""
        if isinstance(v, (list, tuple, set)):
            return ", ".join(str(x) for x in v if x is not None)
        if isinstance(v, dict):
            return ", ".join(str(x) for x in v.values())
        return str(v)

    @field_validator("claim", mode="before")
    @classmethod
    def coerce_claim(cls, v):
        return "" if v is None else str(v)

    @field_validator("verdict", mode="before")
    @classmethod
    def coerce_verdict(cls, v):
        if v is None:
            return "unsupported"
        if isinstance(v, (list, tuple, set)):
            v = next((x for x in v if x is not None), "unsupported")
        return str(v)


class FactualityResult(BaseModel):
    claims: List[FactualClaim] = []
    total_claims: int = 0
    correct_claims: int = 0
    incorrect_claims: int = 0
    unsupported_claims: int = 0
    factuality_score: Optional[float] = None

    @staticmethod
    def _claim_value(claim: Any, *keys: str) -> Any:
        """Read a field from either a dict or a model instance."""
        for key in keys:
            if isinstance(claim, dict):
                val = claim.get(key)
            else:
                val = getattr(claim, key, None)
            if val:
                return val
        return None

    @model_validator(mode="before")
    @classmethod
    def remap_and_recount(cls, data):
        if isinstance(data, dict):
            claims = data.get("claims", []) or []
            n = len(claims)
            data["total_claims"] = n
            if n:
                verdicts = [
                    str(cls._claim_value(c, "verdict", "status", "classification") or "unsupported").lower()
                    for c in claims
                ]
                data["correct_claims"] = sum(1 for v in verdicts if v == "correct")
                data["incorrect_claims"] = sum(1 for v in verdicts if v.startswith("incor"))
                data["unsupported_claims"] = n - data["correct_claims"] - data["incorrect_claims"]
                data["factuality_score"] = min(100.0, round(data["correct_claims"] / n * 100, 1))
            else:
                data["correct_claims"] = 0
                data["incorrect_claims"] = 0
                data["unsupported_claims"] = 0
                data["factuality_score"] = 100.0  # prompt rule: no claims -> 100
        return data


# =====================================================================
# TIER 1 EXTENDED — Faithfulness
# =====================================================================

class Contradiction(BaseModel):
    step_a: int = 0
    step_b: int = 0
    description: str = ""

    @field_validator("step_a", mode="before")
    @classmethod
    def coerce_step_a(cls, v):
        return _coerce_step_int(v)

    @field_validator("step_b", mode="before")
    @classmethod
    def coerce_step_b(cls, v):
        return _coerce_step_int(v)

    @model_validator(mode="before")
    @classmethod
    def remap_llm_fields(cls, data):
        if isinstance(data, dict):
            if "step_a" not in data:
                for alt in ("step1", "first_step", "stepA"):
                    if alt in data:
                        data["step_a"] = data.pop(alt)
                        break
            if "step_b" not in data:
                for alt in ("step2", "second_step", "stepB"):
                    if alt in data:
                        data["step_b"] = data.pop(alt)
                        break
            if "description" not in data:
                for alt in ("desc", "explanation", "detail", "reason", "text"):
                    if alt in data:
                        data["description"] = data.pop(alt)
                        break
        return data


class LogicalGap(BaseModel):
    step_number: int = 0
    description: str = ""

    @field_validator("step_number", mode="before")
    @classmethod
    def coerce_step_number(cls, v):
        return _coerce_step_int(v)

    @model_validator(mode="before")
    @classmethod
    def remap_llm_fields(cls, data):
        if isinstance(data, dict):
            if "step_number" not in data:
                for alt in ("step", "step_id", "step_num", "number"):
                    if alt in data:
                        data["step_number"] = data.pop(alt)
                        break
            if "description" not in data:
                for alt in ("gap", "desc", "explanation", "detail", "reason", "text"):
                    if alt in data:
                        data["description"] = data.pop(alt)
                        break
        return data


class FaithfulnessResult(BaseModel):
    contradictions: List[Contradiction] = []
    conclusion_follows: bool = True
    logical_gaps: List[LogicalGap] = []
    faithfulness_score: int = 3
    summary: str = ""

    @model_validator(mode="before")
    @classmethod
    def remap_llm_fields(cls, data):
        if isinstance(data, dict):
            if "conclusion_follows" not in data:
                for alt in ("conclusion", "follows", "conclusion_supported", "supported"):
                    if alt in data:
                        data["conclusion_follows"] = data.pop(alt)
                        break
            if "faithfulness_score" not in data:
                for alt in ("score", "rating", "likert", "likert_score"):
                    if alt in data:
                        data["faithfulness_score"] = data.pop(alt)
                        break
            if "summary" not in data:
                for alt in ("explanation", "rationale", "assessment", "overall"):
                    if alt in data:
                        data["summary"] = data.pop(alt)
                        break
        return data


# =====================================================================
# TIER 1 EXTENDED — Guideline Adherence (GAR)
# =====================================================================

class GuidelineAdherenceStep(BaseModel):
    step_number: int = 0
    adheres_to_guideline: bool = Field(
        description="True if the step aligns with or does not contradict the guidelines. False only if it violates them.")
    rationale: str = Field(default="", description="Brief explanation of the judgment.")

    @model_validator(mode="before")
    @classmethod
    def remap_llm_fields(cls, data):
        if isinstance(data, dict):
            if "adheres_to_guideline" not in data:
                for alt in ("adheres", "adheres_to_guidelines", "follows_guideline",
                            "is_adherent", "compliant", "value", "result",
                            "judgment", "judgement", "evaluation", "assessment",
                            "verdict", "finding", "determination", "adherence",
                            "complies"):
                    if alt in data:
                        data["adheres_to_guideline"] = data.pop(alt)
                        break
            if "step_number" not in data:
                for alt in ("step", "step_id", "step_num", "number"):
                    if alt in data:
                        data["step_number"] = data.pop(alt)
                        break
        return data

    @field_validator("adheres_to_guideline", mode="before")
    @classmethod
    def coerce_bool(cls, v):
        if isinstance(v, bool):
            return v
        if isinstance(v, dict):
            for key in ("adheres_to_guideline", "adheres", "value", "result"):
                if key in v:
                    v = v[key]
                    break
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ("true", "yes", "1", "adheres", "correct", "compliant"):
                return True
            if s in ("false", "no", "0", "violates", "incorrect", "non_compliant"):
                return False
        try:
            return bool(int(v))
        except (TypeError, ValueError):
            return bool(v)

    @field_validator("step_number", mode="before")
    @classmethod
    def coerce_step_number(cls, v):
        if isinstance(v, dict):
            for alt in ("step", "step_id", "step_num", "number"):
                if alt in v:
                    v = v[alt]
                    break
        try:
            return int(v)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return 0


class GuidelineAdherenceResult(BaseModel):
    steps: List[GuidelineAdherenceStep] = []

    @model_validator(mode="before")
    @classmethod
    def remap_llm_fields(cls, data):
        if isinstance(data, list):
            data = {"steps": data}
        if isinstance(data, dict) and "steps" not in data:
            for alt in ("evaluations", "step_evaluations", "items", "results", "assessments"):
                if alt in data:
                    data["steps"] = data.pop(alt)
                    break
        return data


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
    reasoning_steps: Optional[List[str]] = None
