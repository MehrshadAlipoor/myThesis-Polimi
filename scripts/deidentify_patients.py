#!/usr/bin/env python3
"""
De-identify top-5 patient JSONs for Andrea's benchmark.

Rules:
- Keep patient_id / apollo_id (175-xxx) fully exposed (including in filenames)
- i3lung_id → pseudonym SUBJ_00X (unique per patient, matches GT CSV)
- thread_id, date_logged → "[REDACTED]"
- Remove raw_inputs (real cartella_id 3700006, Italian reports w/ names/dates)
- Remove image_reports
- Remove raw_messages non-tool entries (base64 patient images ~90% size)
- Remove: formatted_report, ml_predictions, step2_*, step3_reasoning, step5_chat_history
- raw_messages: keep ONLY the retrieve_medical_documents tool message
- Filenames: 175-XXX_orchestrator_v0.json (strip UUID = thread_id)
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

SRC_DIR = Path(r"G:\My Drive\00 - Polimi\Thesis\2026_Alipoor_AgentEvalFramework\data\20260915-152734\cases")
DST_DIR = Path(r"G:\My Drive\00 - Polimi\Thesis\2026_Alipoor_AgentEvalFramework\repo\data\deidentified")

# Top-5 patients by max eval time (ordered)
TOP5: List[Tuple[str, str]] = [
    ("175-550", "INT1010747"),
    ("175-911", "INT1010599"),
    ("175-397", "INT1010637"),
    ("175-342", "INT1010066"),
    ("175-149", "INT1010016"),
]

# pseudonym mapping: i3lung_id → SUBJ_00X
PSEUDO = {i3lung: f"SUBJ_{idx:03d}" for idx, (_, i3lung) in enumerate(TOP5, 1)}
# GT mapping: pseudonym → IO_IOCT (from data/data.csv)
GT = {
    "SUBJ_001": 1,
    "SUBJ_002": 1,
    "SUBJ_003": 1,
    "SUBJ_004": 0,
    "SUBJ_005": 0,
}

# orchestrator models we keep (4 per patient)
ORCHESTRATORS = ["baichuan-m2-32b", "gpt-oss-20b", "nemotron-3-nano-30B-A3B", "qwen3-30B-A3B-Thinking"]

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def strip_uuid_from_filename(fname: str) -> str:
    """175-550_baichuan-m2-32b_UUID_v0.json → 175-550_baichuan-m2-32b_v0.json"""
    # pattern: 175-XXX_orchestrator_UUID_v0.json
    parts = fname.split("_")
    if len(parts) >= 4 and parts[0].startswith("175-"):
        # keep patient_id (parts[0]) + orchestrator (parts[1]) + v0.json (last)
        return f"{parts[0]}_{parts[1]}_{parts[-1]}"
    return fname


def redact_thread_id(val: Any) -> str:
    return "[REDACTED]"


def redact_date(val: Any) -> str:
    return "[REDACTED]"


def pseudonymize_i3lung(val: str) -> str:
    return PSEUDO.get(val, val)


def process_json(obj: Any, in_raw_messages: bool = False) -> Any:
    """Recursively process JSON, redacting/removing per rules."""
    if isinstance(obj, dict):
        new_obj = {}
        for k, v in obj.items():
            # Remove entire keys
            if k in (
                "raw_inputs", "image_reports", "formatted_report", "ml_predictions",
                "step2_reasoning", "step2_output", "step2_evaluation",
                "step3_reasoning", "step5_chat_history",
            ):
                continue
            if k == "raw_messages":
                # Keep ONLY tool message with retrieve_medical_documents
                if isinstance(v, list):
                    tool_msgs = []
                    for msg in v:
                        if msg.get("type") == "tool" and msg.get("name") == "retrieve_medical_documents":
                            # Redact its id
                            msg = dict(msg)
                            msg["id"] = "[REDACTED]"
                            tool_msgs.append(msg)
                    if tool_msgs:
                        new_obj[k] = tool_msgs
                continue
            # Redact specific fields
            if k in ("thread_id",):
                new_obj[k] = redact_thread_id(v)
            elif k in ("date_logged",):
                new_obj[k] = redact_date(v)
            elif k in ("i3lung_id",):
                new_obj[k] = pseudonymize_i3lung(str(v))
            elif k == "id" and in_raw_messages:
                new_obj[k] = "[REDACTED]"
            else:
                new_obj[k] = process_json(v, in_raw_messages=(k == "raw_messages"))
        return new_obj
    elif isinstance(obj, list):
        return [process_json(item, in_raw_messages) for item in obj]
    else:
        return obj


def find_patient_files(patient_id: str) -> List[Path]:
    """Find the 4 orchestrator files for a patient."""
    files = []
    for fname in SRC_DIR.iterdir():
        if fname.name.startswith(f"{patient_id}_") and fname.suffix == ".json":
            # Check orchestrator is one of our 4
            for orch in ORCHESTRATORS:
                if f"_{orch}_" in fname.name:
                    files.append(fname)
                    break
    return sorted(files)


def main():
    DST_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing {len(TOP5)} patients × {len(ORCHESTRATORS)} orchestrators = {len(TOP5)*len(ORCHESTRATORS)} files")
    print(f"Source: {SRC_DIR}")
    print(f"Destination: {DST_DIR}")
    print()
    
    total_in = 0
    total_out = 0
    
    for idx, (pid, i3lung) in enumerate(TOP5, 1):
        pseudo = PSEUDO[i3lung]
        print(f"[{idx}/{len(TOP5)}] {pid} ({i3lung}) -> {pseudo}")
        
        files = find_patient_files(pid)
        if len(files) != len(ORCHESTRATORS):
            print(f"  WARNING: expected {len(ORCHESTRATORS)} files, found {len(files)}")
            for f in files:
                print(f"    {f.name}")
        
        for src_f in files:
            total_in += src_f.stat().st_size
            
            with open(src_f, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Process
            data = process_json(data)
            
            # New filename (strip UUID)
            new_fname = strip_uuid_from_filename(src_f.name)
            dst_f = DST_DIR / new_fname
            
            with open(dst_f, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            total_out += dst_f.stat().st_size
            print(f"  {src_f.name} -> {new_fname} ({src_f.stat().st_size/1e6:.1f} MB -> {dst_f.stat().st_size/1024:.1f} KB)")
    
    # Write ground truth CSV
    gt_csv = DST_DIR / "ground_truth.csv"
    with open(gt_csv, "w", encoding="utf-8") as f:
        f.write("Subject;IO_IOCT\n")
        for pseudo, io in GT.items():
            f.write(f"{pseudo};{io}\n")
    print(f"\nGround truth: {gt_csv} ({len(GT)} rows)")
    
    print(f"\nTotal size: {total_in/1e6:.1f} MB -> {total_out/1e6:.1f} MB (reduction: {(1-total_out/total_in)*100:.1f}%)")
    print("Done.")


if __name__ == "__main__":
    main()