"""Data loading: patient records, therapy decisions and ground-truth labels.

Parses the agent-produced patient JSON files produced by the upstream NSCLC
agent pipeline and maps them onto :class:`PatientRecord`, extracts the final
structured therapy decision, loads the ground-truth labels used by the Tier 2
accuracy metric, and resolves the guideline ground truth used by GAR.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from . import config
from .models import PatientRecord


def _normalize_text_value(value: Any) -> str:
    """Flatten arbitrary nested clinical values into a single readable string."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        normalized_items = [_normalize_text_value(item) for item in value]
        return " | ".join(dict.fromkeys(item for item in normalized_items if item))
    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            normalized_item = _normalize_text_value(item)
            if normalized_item:
                parts.append(f"{key}: {normalized_item}")
        return "; ".join(parts)
    return str(value).strip()


def _format_therapy_decision(decision: Any) -> str:
    """Format the structured therapy decision dict into a compact string."""
    if not decision:
        return ""
    if isinstance(decision, str):
        return decision.strip()
    if not isinstance(decision, dict):
        return _normalize_text_value(decision)

    pieces = []
    therapy_class = _normalize_text_value(decision.get("therapy_class"))
    if therapy_class:
        pieces.append(f"therapy_class: {therapy_class}")

    drug_fields = (
        decision.get("exact_drugs")
        or decision.get("drug_names")
        or decision.get("drugs")
        or decision.get("exact_drug")
    )
    drug_text = _normalize_text_value(drug_fields)
    if drug_text:
        pieces.append(f"drugs: {drug_text}")

    reason_text = _normalize_text_value(
        decision.get("reason") or decision.get(
            "rationale") or decision.get("confidence_explanation")
    )
    if reason_text:
        pieces.append(f"reason: {reason_text}")

    confidence_text = _normalize_text_value(decision.get("confidence"))
    if confidence_text:
        pieces.append(f"confidence: {confidence_text}")

    sources_text = _normalize_text_value(decision.get("sources"))
    if sources_text:
        pieces.append(f"sources: {sources_text}")

    return " | ".join(pieces) if pieces else _normalize_text_value(decision)


def _extract_structured_final_decision(data: Dict) -> str:
    """Recover the final structured therapy decision from a patient JSON."""
    pipeline_data = data.get("pipeline_data", {}) or {}
    therapy_decision = pipeline_data.get("therapy_decision", {}) or {}
    structured_decision = therapy_decision.get("best_therapy_reimbursed") or therapy_decision.get(
        "best_therapy_overall"
    )
    if structured_decision:
        return _format_therapy_decision(structured_decision)

    raw_decision = therapy_decision.get("raw_decision")
    if raw_decision:
        return _normalize_text_value(raw_decision)

    step5_logs = pipeline_data.get("step5_logs", [])
    if step5_logs and isinstance(step5_logs[0], dict):
        return _normalize_text_value(step5_logs[0].get("output") or step5_logs[0].get("reasoning"))
    return ""


def load_patient_record(json_file_path: str) -> PatientRecord:
    """Parse a single patient JSON into a :class:`PatientRecord`."""
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pipeline_data = data.get("pipeline_data", {}) or {}
    patient_id = pipeline_data.get("patient_id", "Not found")
    i3lung_id = (
        pipeline_data.get("patient_status", {}).get("i3lung_id")
        or pipeline_data.get("apollo_id")
        or pipeline_data.get("cartella_id")
        or "Not found"
    )
    orchestration_model_name = (
        data.get("config", {}).get("steps", {}).get(
            "step5_decision", {}).get("model") or "Not found"
    )

    raw_steps = pipeline_data.get("therapy_decision", {}).get("reasoning_steps")
    reasoning_steps: List[str] = (
        [_normalize_text_value(s) for s in raw_steps if _normalize_text_value(s)]
        if isinstance(raw_steps, list) else []
    )

    raw_reasoning = _normalize_text_value(raw_steps)
    if not raw_reasoning:
        step5_logs = pipeline_data.get("step5_logs", [{}])
        if step5_logs and isinstance(step5_logs[0], dict):
            raw_reasoning = _normalize_text_value(
                step5_logs[0].get("reasoning"))

    final_decision = _extract_structured_final_decision(data)

    return PatientRecord(
        patient_id=patient_id,
        i3lung_id=i3lung_id,
        orchestration_model_name=orchestration_model_name,
        raw_reasoning=raw_reasoning,
        final_decision=final_decision,
        reasoning_steps=reasoning_steps,
    )


def load_patient_data(json_file_path: str) -> Tuple[PatientRecord, str, Dict]:
    """Load a patient record plus its final decision and raw pipeline data."""
    patient_record = load_patient_record(json_file_path)
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    final_therapy_decision = _extract_structured_final_decision(data)
    pipeline_data = data.get("pipeline_data", {})
    return patient_record, final_therapy_decision, pipeline_data


def load_ground_truth(gt_csv_path: str) -> Dict[str, Any]:
    """Load the ground-truth mapping ``i3lung_id -> IO_IOCT`` from a CSV.

    The CSV is semicolon-delimited and keyed by the ``Subject`` column.
    """
    gt_df = pd.read_csv(gt_csv_path, sep=";",
                        engine="python", on_bad_lines="skip")
    gt_mapping = {}
    for _, row in gt_df.iterrows():
        subject_id = str(row["Subject"]).strip()
        val = row["IO_IOCT"]
        gt_mapping[subject_id] = val
    return gt_mapping


# ---------------------------------------------------------------------
# Guideline ground truth (GAR)
# ---------------------------------------------------------------------

def _format_guidelines_context(gc: Any) -> str:
    """Flatten the structured ``guidelines_context`` dict into readable text."""
    if not isinstance(gc, dict):
        return _normalize_text_value(gc)
    parts = []
    for key, label in [("all_sources", "ALL SOURCES CONSULTED"),
                       ("recommended_therapies", "RECOMMENDED THERAPIES"),
                       ("excluded_therapies", "EXCLUDED THERAPIES"),
                       ("raw_analysis", "ANALYSIS")]:
        val = _normalize_text_value(gc.get(key))
        if val:
            parts.append(f"{label}: {val}")
    return "\n".join(parts)


def _extract_retrieved_guidelines(pipeline_data: Optional[Dict]) -> str:
    """Legacy fallback: verbatim guideline PDF text from raw_messages tool calls."""
    if not pipeline_data:
        return ""
    chunks = []
    for msg in pipeline_data.get("raw_messages", []):
        if isinstance(msg, dict) and msg.get("type") == "tool" and msg.get("name") == "retrieve_medical_documents":
            content = msg.get("content", "")
            if content:
                chunks.append(str(content))
    return "\n\n".join(chunks)


def extract_guideline_ground_truth(pipeline_data: Optional[Dict]) -> Tuple[str, str]:
    """Priority chain for GAR ground truth (fixes the ~14% missing GAR):

    1. ``guidelines_context`` (structured, most reliable)
    2. ``step3_output`` (compiled guideline text)
    3. ``raw_messages`` retrieve_medical_documents tool content

    Returns ``(text_capped_at_GUIDELINE_CHAR_CAP, source)`` where source is one
    of ``{"guidelines_context", "step3_output", "raw_messages", "none"}``.
    """
    pd_ = pipeline_data or {}
    gc_text = _format_guidelines_context(pd_.get("guidelines_context")).strip()
    if gc_text:
        return gc_text[:config.GUIDELINE_CHAR_CAP], "guidelines_context"
    so_text = str(pd_.get("step3_output") or "").strip()
    if so_text:
        return so_text[:config.GUIDELINE_CHAR_CAP], "step3_output"
    rm_text = _extract_retrieved_guidelines(pd_).strip()
    if rm_text:
        return rm_text[:config.GUIDELINE_CHAR_CAP], "raw_messages"
    return "", "none"
