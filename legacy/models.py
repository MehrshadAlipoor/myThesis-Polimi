from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


# --- Layer 1: MedR-Bench Reasoning Models ---
class StepCategory(str, Enum):
    CITATION = "Citation"   # Restating facts
    REASONING = "Reasoning"  # Applying medical logic
    REDUNDANCY = "Redundancy"  # Fluff/Repetition
    REPETITION = "Repetition"  # Repeating


class AtomicStep(BaseModel):
    step_id: int
    claim: str
    category: StepCategory
    factuality_keywords: List[str] = Field(description="Search terms to verify this claim.")


class ReasoningQualityResponse(BaseModel):
    steps: List[AtomicStep]


# --- Layer 2: NOHARM Safety Models ---
class HarmSeverity(str, Enum):
    NONE = "None"
    MILD = "Mild"
    MODERATE = "Moderate"
    SEVERE = "Severe"


class ActionEvaluation(BaseModel):
    recommendation: str
    error_type: str = Field(description="Commission, Omission, or None")
    is_valid_alternative: bool
    severity: HarmSeverity
    effectiveness_score: float = Field(ge=0, le=1)
    rationale: str


class SafteyReport(BaseModel):
    evaluations: List[ActionEvaluation]
