"""Utilities for file indexing and evaluation result persistence."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from . import config


def _sanitize_filename(value: Any) -> str:
    """Replace characters that are invalid on common filesystems."""
    return (
        str(value)
        .replace(":", "-")
        .replace("/", "-")
        .replace("\\", "-")
        .replace("*", "-")
        .replace("?", "-")
        .replace('"', "-")
        .replace("<", "-")
        .replace(">", "-")
        .replace("|", "-")
    )


def _get_orchestrator_model_from_file(json_file_path: str) -> str:
    """Read the orchestrator model name embedded in a patient JSON."""
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("config", {}).get("steps", {}).get("step5_decision", {}).get("model", "Unknown")


def build_file_index(data_dir: str) -> Tuple[Dict[str, List[str]], List[str]]:
    """Index patient JSON files grouped by orchestrator model."""
    orch_to_files = defaultdict(list)
    for root, _, files in os.walk(data_dir):
        for fname in sorted(files):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(root, fname)
            orch_model = _get_orchestrator_model_from_file(fpath)
            orch_to_files[orch_model].append(fpath)
    orchestrators = sorted(orch_to_files.keys())
    print(
        f"Indexed {sum(len(v) for v in orch_to_files.values())} files "
        f"across {len(orchestrators)} orchestrators: {orchestrators}"
    )
    return dict(orch_to_files), orchestrators


def _get_run_output_dir(run_date: str, run_timestamp: str, judge_model: Optional[str] = None) -> str:
    active_judge_model = judge_model or config.JUDGE_MODEL
    return f"eval/{run_date}/{run_timestamp}/{_sanitize_filename(active_judge_model)}"


def _write_error_log(error_log_path: str, error_records: List[Dict]) -> None:
    os.makedirs(os.path.dirname(error_log_path), exist_ok=True)
    with open(error_log_path, "w", encoding="utf-8") as f:
        json.dump(error_records, f, indent=4, ensure_ascii=False)


def save_evaluation_results(
    evaluation_data: Dict,
    patient_id: str,
    run_date: str,
    run_timestamp: str,
    orchestrator_model: Optional[str] = None,
    judge_model: Optional[str] = None,
) -> str:
    """Persist a single patient evaluation result to ``eval/<date>/<time>/<judge>/``."""
    orchestration_model_name = orchestrator_model or "unknown_orchestrator"
    dir_path = _get_run_output_dir(run_date, run_timestamp, judge_model=judge_model)
    os.makedirs(dir_path, exist_ok=True)
    file_path = f"{dir_path}/{patient_id}_evaluation_{_sanitize_filename(orchestration_model_name)}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_data, f, indent=4, ensure_ascii=False)
    return file_path
