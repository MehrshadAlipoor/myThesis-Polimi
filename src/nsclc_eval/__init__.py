"""NSCLC therapy-agent evaluation framework.

A multi-tier LLM-as-a-Judge framework for assessing AI agents that generate
first-line NSCLC (non-small-cell lung cancer) therapy recommendations.

Tiers
-----
* **Tier 1 Core** — Reasoning Efficiency Score (RES): decomposes reasoning
  into atomic steps and classifies each as Citation/Repetition/Reasoning/Redundancy.
* **Tier 1 Extended** — Reasoning Completeness, Factuality, Faithfulness and
  Guideline Adherence Ratio (GAR).
* **Tier 2** — Binary clinical accuracy (IO vs IOCT) and Treatment Plan
  Completeness.

The judge and orchestrator models are open-weight LLMs served through a local
``llama-cpp-server`` Docker container (CUDA-accelerated).
"""

from . import config, data_loading, models, pipeline, prompts
from .data_loading import (
    PatientRecord,
    load_ground_truth,
    load_patient_data,
    load_patient_record,
)
from .pipeline import (
    DEFAULT_EVAL_FLAGS,
    run_all_patient_evaluations,
    run_evaluation,
)
from .tier1_core import (
    calculate_res,
    classify_reasoning_steps,
    decompose_reasoning,
)
from .tier1_extended import (
    evaluate_factuality,
    evaluate_faithfulness,
    evaluate_guideline_adherence,
    evaluate_reasoning_completeness_llm,
    evaluate_treatment_plan_completeness,
)
from .tier2_clinical import evaluate_clinical_effectiveness
from .utils import build_file_index, save_evaluation_results

__all__ = [
    "config",
    "data_loading",
    "models",
    "pipeline",
    "prompts",
    "PatientRecord",
    "load_ground_truth",
    "load_patient_data",
    "load_patient_record",
    "DEFAULT_EVAL_FLAGS",
    "run_all_patient_evaluations",
    "run_evaluation",
    "calculate_res",
    "classify_reasoning_steps",
    "decompose_reasoning",
    "evaluate_factuality",
    "evaluate_faithfulness",
    "evaluate_guideline_adherence",
    "evaluate_reasoning_completeness_llm",
    "evaluate_treatment_plan_completeness",
    "evaluate_clinical_effectiveness",
    "build_file_index",
    "save_evaluation_results",
]

__version__ = "0.1.0"
