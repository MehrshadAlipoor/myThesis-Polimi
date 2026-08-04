"""Tier 1 Extended — Reasoning Completeness, Factuality, Faithfulness, GAR.

Each metric is a single LLM-as-a-Judge call using structured
``response_format`` parsing. The completeness metrics score an 8-item clinical
checklist; factuality verifies claims against the patient record; faithfulness
rates internal consistency (Likert 1-5); and the Guideline Adherence Ratio
(GAR) checks every reasoning step against retrieved ESMO/ASCO guidelines.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional

from . import config
from .models import (
    FactualityResult,
    FaithfulnessResult,
    GuidelineAdherenceResult,
    ReasoningCompletenessExtraction,
)
from .prompts import (
    FACTUALITY_PROMPT,
    FAITHFULNESS_PROMPT,
    GUIDELINE_ADHERENCE_PROMPT,
    REASONING_COMPLETENESS_PROMPT,
    TREATMENT_PLAN_COMPLETENESS_PROMPT,
)


def _extract_retrieved_guidelines(pipeline_data: Optional[Dict]) -> str:
    """Extract verbatim guideline PDF text from raw_messages (ground truth for GAR)."""
    if not pipeline_data:
        return ""
    chunks = []
    for msg in pipeline_data.get("raw_messages", []):
        if msg.get("type") == "tool" and msg.get("name") == "retrieve_medical_documents":
            content = msg.get("content", "")
            if content:
                chunks.append(str(content))
    return "\n\n".join(chunks)


def _flatten_patient_status(patient_status: Optional[Dict]) -> Dict:
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


# === METRIC 1: Reasoning / Treatment Plan Completeness (LLM) ===

def _run_completeness_eval(
    text_data: str, pipeline_data: Dict, prompt_template: str
) -> Dict:
    """Run a single completeness evaluation over an 8-item clinical checklist."""
    patient_status = pipeline_data.get("patient_status", {})
    flat_ps = _flatten_patient_status(patient_status)
    concise_patient = json.dumps(
        flat_ps, indent=2, default=str, ensure_ascii=False)

    user_message = f"{text_data}\n\nPATIENT_DATA:\n{concise_patient}"

    try:
        response = config.client.beta.chat.completions.parse(
            model=config.JUDGE_MODEL,
            messages=[
                {"role": "system", "content": prompt_template},
                {"role": "user", "content": user_message},
            ],
            response_format=ReasoningCompletenessExtraction,
            temperature=config.TEMPERATURE,
            seed=config.SEED,
        )
        result = response.choices[0].message.parsed

        n_mentioned = sum(1 for item in result.items if item.mentioned)
        completeness_score = round(n_mentioned / 8 * 100, 1)

        missing = [
            item.checklist_item for item in result.items if not item.mentioned]

        result_dict = result.model_dump()
        result_dict["completeness_score"] = completeness_score
        result_dict["missing_fields"] = missing

        print(
            f"  Completeness: {completeness_score}% ({n_mentioned}/8 items mentioned)")
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
    return _run_completeness_eval(
        f"REASONING_STEPS:\n{steps_text}", pipeline_data, REASONING_COMPLETENESS_PROMPT
    )


def evaluate_treatment_plan_completeness(treatment_plan_text: str, pipeline_data: Dict) -> Dict:
    """Score how many of the 8 clinical factors appear in the final treatment plan."""
    print("\n=== Evaluating Treatment Plan Completeness (Tier 2) ===")
    return _run_completeness_eval(
        f"TREATMENT_PLAN:\n{treatment_plan_text}", pipeline_data, TREATMENT_PLAN_COMPLETENESS_PROMPT
    )


# === METRIC 2: Factuality (1 LLM call) ===

def evaluate_factuality(steps: List[str], pipeline_data: Dict) -> Dict:
    """Verify the factual claims in the reasoning steps against patient data."""
    patient_status = pipeline_data.get("patient_status", {})
    flat_ps = _flatten_patient_status(patient_status)
    concise_patient = json.dumps(
        flat_ps, indent=2, default=str, ensure_ascii=False)
    steps_text = "\n".join(f"Step {i+1}: {s}" for i, s in enumerate(steps))
    user_message = f"PATIENT_DATA:\n{concise_patient}\n\nREASONING_STEPS:\n{steps_text}"

    try:
        response = config.client.beta.chat.completions.parse(
            model=config.JUDGE_MODEL,
            messages=[
                {"role": "system", "content": FACTUALITY_PROMPT},
                {"role": "user", "content": user_message},
            ],
            response_format=FactualityResult,
            temperature=config.TEMPERATURE,
            seed=config.SEED,
        )
        result = response.choices[0].message.parsed
        result_dict = result.model_dump()
        print(
            f"  Factuality: {result_dict['factuality_score']:.1f}% "
            f"({result_dict['correct_claims']}/{result_dict['total_claims']} correct, "
            f"{result_dict['incorrect_claims']} incorrect, "
            f"{result_dict['unsupported_claims']} unsupported)"
        )
        return result_dict
    except Exception as e:
        print(f"  Factuality ERROR: {e}")
        return {
            "claims": [], "total_claims": 0, "correct_claims": 0,
            "incorrect_claims": 0, "unsupported_claims": 0, "factuality_score": 0.0,
            "error": str(e),
        }


# === METRIC 3: Faithfulness (1 LLM call) ===

def evaluate_faithfulness(steps: List[str], raw_decision: str) -> Dict:
    """Rate internal consistency of the reasoning chain (Likert 1-5)."""
    steps_text = "\n".join(f"Step {i+1}: {s}" for i, s in enumerate(steps))
    user_message = f"REASONING_STEPS:\n{steps_text}\n\nFINAL_RECOMMENDATION:\n{raw_decision[:2000]}"

    try:
        response = config.client.beta.chat.completions.parse(
            model=config.JUDGE_MODEL,
            messages=[
                {"role": "system", "content": FAITHFULNESS_PROMPT},
                {"role": "user", "content": user_message},
            ],
            response_format=FaithfulnessResult,
            temperature=config.TEMPERATURE,
            seed=config.SEED,
        )
        result = response.choices[0].message.parsed
        result_dict = result.model_dump()
        print(
            f"  Faithfulness: {result_dict['faithfulness_score']}/5 "
            f"| Contradictions: {len(result_dict['contradictions'])} "
            f"| Conclusion follows: {result_dict['conclusion_follows']}"
        )
        return result_dict
    except Exception as e:
        print(f"  Faithfulness ERROR: {e}")
        return {
            "contradictions": [], "conclusion_follows": True,
            "logical_gaps": [], "faithfulness_score": 0,
            "summary": f"Error: {e}", "error": str(e),
        }


# === METRIC 4: Guideline Adherence Ratio (GAR) — 1 LLM call ===

def evaluate_guideline_adherence(steps: List[str], step3_output: str) -> Dict:
    """Check each reasoning step against the retrieved clinical guidelines."""
    print("\n=== Evaluating Guideline Adherence (Tier 1) ===")
    steps_text = "\n".join(f"Step {i+1}: {s}" for i, s in enumerate(steps))
    user_message = f"GUIDELINES_GROUND_TRUTH:\n{step3_output}\n\nREASONING_STEPS:\n{steps_text}"

    try:
        response = config.client.beta.chat.completions.parse(
            model=config.JUDGE_MODEL,
            messages=[
                {"role": "system", "content": GUIDELINE_ADHERENCE_PROMPT},
                {"role": "user", "content": user_message},
            ],
            response_format=GuidelineAdherenceResult,
            temperature=config.TEMPERATURE,
            seed=config.SEED,
        )

        result = response.choices[0].message.parsed
        total_steps = len(result.steps)
        if total_steps == 0:
            return {"gar_score": 0.0, "details": []}

        adherent_steps = sum(
            1 for step in result.steps if step.adheres_to_guideline)
        gar_score = round((adherent_steps / total_steps) * 100, 1)

        print(f"  GAR: {gar_score}% ({adherent_steps}/{total_steps})")
        return {
            "gar_score": gar_score,
            "total_steps": total_steps,
            "adherent_steps": adherent_steps,
            "details": [s.model_dump() for s in result.steps],
        }
    except Exception as e:
        print(f"  GAR ERROR: {e}")
        return {"gar_score": 0.0, "error": str(e), "details": []}
