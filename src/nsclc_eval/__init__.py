"""NSCLC therapy-agent evaluation framework.

A multi-tier LLM-as-a-Judge framework for assessing AI agents that generate
first-line NSCLC (non-small-cell lung cancer) therapy recommendations.

Tiers
-----
* **Tier 1 Core** — Reasoning Efficiency Score (RES): decomposes reasoning
  into atomic steps and classifies each as Citation/Repetition/Reasoning/Redundancy
  in a single batched call.
* **Tier 1 Extended** — Reasoning Completeness, Factuality, Faithfulness and
  Guideline Adherence Ratio (GAR).
* **Tier 2** — Binary clinical accuracy (IO vs IOCT), Treatment Plan
  Completeness and Output Factuality.

Extras
------
* **Speed edition** — per-metric ``max_tokens`` caps, a per-request
  ``thinking_budget_tokens`` cap for the thinking judges, and optional reuse of
  prompt-invariant metrics.
* **Prompt sensitivity** — PSS/JSS across paraphrased judge prompts
  (:mod:`nsclc_eval.sensitivity`).

The judge and orchestrator models are open-weight LLMs served through a local
``llama-cpp-server`` Docker container (CUDA-accelerated).
"""

from . import config, data_loading, llm, models, pipeline, prompts, sensitivity
from .data_loading import (
    PatientRecord,
    extract_guideline_ground_truth,
    load_ground_truth,
    load_patient_data,
    load_patient_record,
)
from .llm import create_call, parse_call
from .pipeline import (
    DEFAULT_EVAL_FLAGS,
    run_all_patient_evaluations,
    run_evaluation,
)
from .sensitivity import (
    PROMPT_VARIANT_SETS,
    compute_sensitivity_metrics,
    run_sensitivity_experiment,
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
    evaluate_output_factuality,
    evaluate_reasoning_completeness_llm,
    evaluate_treatment_plan_completeness,
)
from .tier2_clinical import evaluate_clinical_effectiveness
from .utils import build_file_index, save_evaluation_results

__all__ = [
    "config",
    "data_loading",
    "llm",
    "models",
    "pipeline",
    "prompts",
    "sensitivity",
    "PatientRecord",
    "extract_guideline_ground_truth",
    "load_ground_truth",
    "load_patient_data",
    "load_patient_record",
    "create_call",
    "parse_call",
    "DEFAULT_EVAL_FLAGS",
    "run_all_patient_evaluations",
    "run_evaluation",
    "PROMPT_VARIANT_SETS",
    "compute_sensitivity_metrics",
    "run_sensitivity_experiment",
    "calculate_res",
    "classify_reasoning_steps",
    "decompose_reasoning",
    "evaluate_factuality",
    "evaluate_faithfulness",
    "evaluate_guideline_adherence",
    "evaluate_output_factuality",
    "evaluate_reasoning_completeness_llm",
    "evaluate_treatment_plan_completeness",
    "evaluate_clinical_effectiveness",
    "build_file_index",
    "save_evaluation_results",
]

__version__ = "0.2.0"
