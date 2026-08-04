"""Data loading: patient records, therapy decisions and ground-truth labels.

Parses the agent-produced patient JSON files produced by the upstream
NSCLC agent pipeline and maps them onto :class:`PatientRecord`, extracts the
final structured therapy decision, and loads the ground-truth labels used by
the Tier 2 accuracy metric.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Tuple

import pandas as pd

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

    raw_reasoning = _normalize_text_value(pipeline_data.get(
        "therapy_decision", {}).get("reasoning_steps"))
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
