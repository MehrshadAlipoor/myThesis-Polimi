"""Tier 1 Extended — Reasoning Completeness, Factuality, Faithfulness, GAR.

Each metric is a single LLM-as-a-Judge call using structured
``response_format`` parsing. Completeness scores an 8-item clinical checklist;
factuality verifies claims against the patient record; faithfulness rates
internal consistency (Likert 1-5); and the Guideline Adherence Ratio (GAR)
checks every reasoning step against retrieved ESMO/ASCO guidelines.

Output is capped per metric (``config.MAX_TOKENS``) and routed through the
shared LLM infrastructure (``nsclc_eval.llm``), which also applies the
per-request thinking budget for the thinking judges.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from . import config
from .llm import parse_call
from .models import (
    FactualityResult,
    FaithfulnessResult,
    GuidelineAdherenceResult,
    ReasoningCompletenessExtraction,
)
from .prompts import P

_CANONICAL_FIELDS = [
    "histology", "pd_l1", "ecog_ps", "comorbidities",
    "disease_burden", "io_biomarkers", "stage", "ct_io_drug_contraindications",
]


def _flatten_patient_status(patient_status: Dict | None) -> Dict:
    """Flatten the patient_status dict to a simple key-value mapping for LLM input."""
    if not patient_status:
        return {}
    flat = {}
    for key, val in patient_status.items():
        if key in ("data_quality", "loris_features_from_db"):
            continue
        if isinstance(val, list) and len(val) > 0 and isinstance(val[0], dict) and "value" in val[0]:
            flat[key] = val[0]["value"]
        elif isinstance(val, dict) and "value" in val:
            flat[key] = val["value"]
        else:
            flat[key] = val
    return flat


def _concise_patient_data(pipeline_data: Dict) -> str:
    flat = _flatten_patient_status(pipeline_data.get("patient_status", {}))
    return json.dumps(flat, indent=2, default=str, ensure_ascii=False)[:config.PATIENT_DATA_CHAR_CAP]


def _to_canonical(name: str) -> str | None:
    n = re.sub(r"[^a-z0-9]", "", str(name).lower())
    if not n:
        return None
    if "histolog" in n:
        return "histology"
    if "pdl" in n:
        return "pd_l1"
    if "ecog" in n or "egoc" in n or "performance" in n:
        return "ecog_ps"
    if "comorbid" in n:
        return "comorbidities"
    if "burden" in n:
        return "disease_burden"
    if "biomarker" in n:
        return "io_biomarkers"
    if "stage" in n or "tnm" in n:
        return "stage"
    if "contraindication" in n:
        return "ct_io_drug_contraindications"
    return None


# === METRIC 1 & 2: Reasoning / Treatment Plan Completeness (LLM) ===

def _run_completeness_eval(text_data: str, pipeline_data: Dict, prompt_template: str,
                           max_tokens_key: str = "completeness") -> Dict:
    """Run a single completeness evaluation over the 8-item clinical checklist."""
    user_message = f"{text_data}\n\nPATIENT_DATA:\n{_concise_patient_data(pipeline_data)}"
    try:
        response = parse_call(
            model=config.JUDGE_MODEL,
            messages=[
                {"role": "system", "content": prompt_template},
                {"role": "user", "content": user_message},
            ],
            response_format=ReasoningCompletenessExtraction,
            temperature=config.TEMPERATURE, seed=config.SEED,
            max_tokens=config.MAX_TOKENS[max_tokens_key],
        )
        result = response.choices[0].message.parsed
        items = result.items if result else []

        # Canonical-8 completeness: count only the 8 required clinical factors.
        # Extra/non-canonical items are ignored; duplicates count once.
        mentioned = set()
        for item in items:
            if item.mentioned:
                c = _to_canonical(item.checklist_item)
                if c:
                    mentioned.add(c)
        n_mentioned = len(mentioned)
        completeness_score = min(100.0, round(n_mentioned / 8 * 100, 1))
        missing = [c for c in _CANONICAL_FIELDS if c not in mentioned]

        result_dict: Dict[str, Any] = result.model_dump() if result else {"items": []}
        result_dict["completeness_score"] = completeness_score
        result_dict["missing_fields"] = missing
        print(f"  Completeness: {completeness_score}% ({n_mentioned}/8 items mentioned)")
        if missing:
            print(f"  Missing: {missing}")
        return result_dict
    except Exception as e:
        print(f"  Completeness ERROR: {e}")
        return {"items": [], "completeness_score": 0.0, "error": str(e), "missing_fields": []}


def evaluate_reasoning_completeness_llm(steps: List[str], pipeline_data: Dict) -> Dict:
    """Score how many of the 8 clinical factors appear in the reasoning steps."""
    print("\n=== Evaluating Reasoning Completeness (Tier 1) ===")
    steps_text = "\n".join(f"Step {i+1}: {s}" for i, s in enumerate(steps))
    return _run_completeness_eval(f"REASONING_STEPS:\n{steps_text}", pipeline_data,
                                  P("REASONING_COMPLETENESS_PROMPT"), max_tokens_key="completeness")


def evaluate_treatment_plan_completeness(treatment_plan_text: str, pipeline_data: Dict) -> Dict:
    """Score how many of the 8 clinical factors appear in the final treatment plan."""
    print("\n=== Evaluating Treatment Plan Completeness (Tier 2) ===")
    return _run_completeness_eval(f"TREATMENT_PLAN:\n{treatment_plan_text}", pipeline_data,
                                  P("TREATMENT_PLAN_COMPLETENESS_PROMPT"), max_tokens_key="treat_comp")


# === METRIC 3: Factuality (1 LLM call) ===

def evaluate_factuality(steps: List[str], pipeline_data: Dict) -> Dict:
    """Verify the factual claims in the reasoning steps against patient data."""
    steps_text = "\n".join(f"Step {i+1}: {s}" for i, s in enumerate(steps))
    user_message = f"PATIENT_DATA:\n{_concise_patient_data(pipeline_data)}\n\nREASONING_STEPS:\n{steps_text}"
    try:
        response = parse_call(
            model=config.JUDGE_MODEL,
            messages=[
                {"role": "system", "content": P("FACTUALITY_PROMPT")},
                {"role": "user", "content": user_message},
            ],
            response_format=FactualityResult,
            temperature=config.TEMPERATURE, seed=config.SEED,
            max_tokens=config.MAX_TOKENS["factuality"],
        )
        result = response.choices[0].message.parsed
        result_dict = result.model_dump() if result else {}
        fs = result_dict.get("factuality_score")
        if fs is None:
            print("  Factuality: N/A (judge returned no claims -> not evaluated)")
        else:
            print(f"  Factuality: {fs:.1f}% "
                  f"({result_dict.get('correct_claims', 0)}/{result_dict.get('total_claims', 0)} correct, "
                  f"{result_dict.get('incorrect_claims', 0)} incorrect, "
                  f"{result_dict.get('unsupported_claims', 0)} unsupported)")
        return result_dict
    except Exception as e:
        print(f"  Factuality ERROR: {e}")
        return {"claims": [], "total_claims": 0, "correct_claims": 0,
                "incorrect_claims": 0, "unsupported_claims": 0, "factuality_score": None,
                "error": str(e)}


def evaluate_output_factuality(output_text: str, pipeline_data: Dict) -> Dict:
    """Verify the factual claims in the agent's final output against patient data."""
    print("\n=== Evaluating Output Factuality (Tier 2) ===")
    user_message = (f"FINAL_OUTPUT:\n{output_text[:config.PATIENT_DATA_CHAR_CAP]}\n\n"
                    f"PATIENT_DATA:\n{_concise_patient_data(pipeline_data)}")
    try:
        response = parse_call(
            model=config.JUDGE_MODEL,
            messages=[
                {"role": "system", "content": P("OUTPUT_FACTUALITY_PROMPT")},
                {"role": "user", "content": user_message},
            ],
            response_format=FactualityResult,
            temperature=config.TEMPERATURE, seed=config.SEED,
            max_tokens=config.MAX_TOKENS["out_fact"],
        )
        result = response.choices[0].message.parsed
        result_dict = result.model_dump() if result else {}
        fs = result_dict.get("factuality_score")
        if fs is None:
            print("  Output Factuality: N/A (judge returned no claims -> not evaluated)")
        else:
            print(f"  Output Factuality: {fs:.1f}% "
                  f"({result_dict.get('correct_claims', 0)}/{result_dict.get('total_claims', 0)} correct)")
        return result_dict
    except Exception as e:
        print(f"  Output Factuality ERROR: {e}")
        return {"claims": [], "total_claims": 0, "correct_claims": 0,
                "incorrect_claims": 0, "unsupported_claims": 0, "factuality_score": None,
                "error": str(e)}


# === METRIC 4: Faithfulness (1 LLM call) ===

def evaluate_faithfulness(steps: List[str], raw_decision: str) -> Dict:
    """Rate internal consistency of the reasoning chain (Likert 1-5)."""
    steps_text = "\n".join(f"Step {i+1}: {s}" for i, s in enumerate(steps))
    user_message = f"REASONING_STEPS:\n{steps_text}\n\nFINAL_RECOMMENDATION:\n{raw_decision[:2000]}"
    try:
        response = parse_call(
            model=config.JUDGE_MODEL,
            messages=[
                {"role": "system", "content": P("FAITHFULNESS_PROMPT")},
                {"role": "user", "content": user_message},
            ],
            response_format=FaithfulnessResult,
            temperature=config.TEMPERATURE, seed=config.SEED,
            max_tokens=config.MAX_TOKENS["faithfulness"],
        )
        result = response.choices[0].message.parsed
        result_dict = result.model_dump() if result else {}
        print(f"  Faithfulness: {result_dict.get('faithfulness_score', 0)}/5 "
              f"| Contradictions: {len(result_dict.get('contradictions', []))} "
              f"| Conclusion follows: {result_dict.get('conclusion_follows')}")
        return result_dict
    except Exception as e:
        print(f"  Faithfulness ERROR: {e}")
        return {"contradictions": [], "conclusion_follows": True,
                "logical_gaps": [], "faithfulness_score": 0,
                "summary": f"Error: {e}", "error": str(e)}


# === METRIC 5: Guideline Adherence Ratio (GAR) — 1 LLM call ===

def evaluate_guideline_adherence(steps: List[str], guideline_ground_truth_text: str) -> Dict:
    """Check each reasoning step against the retrieved clinical guidelines.

    ``gar_status`` ∈ {"evaluated", "no_guideline_ground_truth",
    "judge_returned_no_steps", "error"}; ``gar_score`` is None whenever the
    metric could not be evaluated (never conflated with a genuine 0%).
    """
    print("\n=== Evaluating Guideline Adherence (Tier 1) ===")
    n_input_steps = len(steps)
    if not guideline_ground_truth_text:
        print("  SKIPPED: no guideline ground truth available for this patient.")
        return {"gar_score": None, "gar_status": "no_guideline_ground_truth",
                "total_steps": n_input_steps, "adherent_steps": 0, "details": []}

    steps_text = "\n".join(f"Step {i+1}: {s}" for i, s in enumerate(steps))
    user_message = f"GUIDELINES_GROUND_TRUTH:\n{guideline_ground_truth_text}\n\nREASONING_STEPS:\n{steps_text}"
    try:
        response = parse_call(
            model=config.JUDGE_MODEL,
            messages=[
                {"role": "system", "content": P("GUIDELINE_ADHERENCE_PROMPT")},
                {"role": "user", "content": user_message},
            ],
            response_format=GuidelineAdherenceResult,
            temperature=config.TEMPERATURE, seed=config.SEED,
            max_tokens=config.MAX_TOKENS["gar"],
        )
        result = response.choices[0].message.parsed
        judged_steps = result.steps if result else []
        total_steps = len(judged_steps)
        if total_steps == 0:
            print("  WARNING: judge returned zero step evaluations.")
            return {"gar_score": None, "gar_status": "judge_returned_no_steps",
                    "total_steps": n_input_steps, "adherent_steps": 0, "details": []}
        adherent_steps = sum(1 for s in judged_steps if s.adheres_to_guideline)
        gar_score = round((adherent_steps / total_steps) * 100, 1)
        print(f"  GAR: {gar_score}% ({adherent_steps}/{total_steps})")
        return {"gar_score": gar_score, "gar_status": "evaluated",
                "total_steps": total_steps, "adherent_steps": adherent_steps,
                "details": [s.model_dump() for s in judged_steps]}
    except Exception as e:
        print(f"  GAR ERROR: {e}")
        return {"gar_score": None, "gar_status": "error", "error": str(e),
                "total_steps": n_input_steps, "adherent_steps": 0, "details": []}
