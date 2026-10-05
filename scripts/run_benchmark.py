#!/usr/bin/env python3
"""
Turnkey benchmark for Andrea's A40.

Runs the NSCLC evaluation framework on the de-identified cohort (5 patients × 4 orchestrators = 20 files)
with the 7 core metrics (GAR excluded) and reports per-judge per-patient latency.

This script uses the existing llama-cpp-server stack (no vLLM changes).

Usage:
    python scripts/run_benchmark.py [--data-dir DIR] [--gt CSV] [--judges JUDGE ...] [--max-files N] [--no-restart]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure src/ is on sys.path for the nsclc_eval package
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nsclc_eval import config
from nsclc_eval.data_loading import load_ground_truth
from nsclc_eval.pipeline import run_all_patient_evaluations


# 7 core metrics for the benchmark (GAR / tier1_guideline_adherence explicitly OFF)
BENCHMARK_EVAL_FLAGS = {
    "tier1_core": True,
    "tier1_reasoning_completeness": True,
    "tier1_guideline_adherence": False,  # GAR dropped for this benchmark
    "tier1_factuality": True,
    "tier1_faithfulness": True,
    "tier2_binary_accuracy": True,
    "tier2_treatment_completeness": True,
    "tier2_factuality": True,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_benchmark",
        description="Run the NSCLC judge latency benchmark (7 metrics, GAR off) on the de-identified cohort.",
    )
    parser.add_argument(
        "--data-dir",
        default=str(REPO_ROOT / "data" / "deidentified"),
        help="Directory containing patient JSON files (default: data/deidentified)",
    )
    parser.add_argument(
        "--gt",
        default=str(REPO_ROOT / "data" / "deidentified" / "ground_truth.csv"),
        help="Path to ground-truth CSV (default: data/deidentified/ground_truth.csv)",
    )
    parser.add_argument(
        "--judges",
        nargs="+",
        default=None,
        help="Explicit judge model names to evaluate (default: all models from models.ini except bge/medgemma)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum patient files per (judge, orchestrator) batch (default: all)",
    )
    parser.add_argument(
        "--no-restart",
        action="store_true",
        help="Do NOT restart the llama-cpp-server container between judges (assumes external server at NSCLC_EVAL_SERVER_URL)",
    )
    parser.add_argument(
        "--orchestrators",
        nargs="+",
        default=None,
        help="Explicit orchestrator model names to evaluate (default: all discovered from data-dir)",
    )
    return parser.parse_args()


def load_gt(gt_path: str):
    """Load ground-truth mapping from CSV."""
    print(f"Loading ground truth from {gt_path}...")
    gt = load_ground_truth(gt_path)
    print(f"  {len(gt)} patients in ground truth")
    return gt


def run_benchmark(
    data_dir: str,
    gt_mapping: dict,
    judge_models: list[str] | None,
    max_files: int | None,
    no_restart: bool,
    orchestrators: list[str] | None,
) -> list:
    """Run the full evaluation matrix and return results."""
    print(f"\nData dir: {data_dir}")
    print(f"Judges: {judge_models or '(all from models.ini)'}")
    print(f"Orchestrators: {orchestrators or '(all discovered from data-dir)'}")
    print(f"Max files per batch: {max_files or 'all'}")
    print(f"Restart server between judges: {not no_restart}")
    print(f"Metrics: {sorted(k for k, v in BENCHMARK_EVAL_FLAGS.items() if v)}")
    print(f"  (GAR / tier1_guideline_adherence: OFF)")
    print("-" * 60)

    t_start = time.time()
    results = run_all_patient_evaluations(
        data_dir=data_dir,
        gt_mapping=gt_mapping,
        judge_models=judge_models,
        mode="both",
        orchestrators_to_evaluate=orchestrators,
        max_files_per_batch=max_files,
        save_results=True,
        verbose=True,
        eval_flags=BENCHMARK_EVAL_FLAGS,
        restart_server=not no_restart,
    )
    t_total = time.time() - t_start

    print(f"\n{'='*60}")
    print(f"BENCHMARK COMPLETED in {t_total/60:.1f} min")
    print(f"Total evaluations: {len(results)}")
    print(f"{'='*60}")

    return results


def summarize_latency(results: list) -> None:
    """Print per-judge × per-patient latency table from saved evaluation JSONs."""
    rows = []
    for r in results:
        saved = r.get("saved_path")
        if not saved:
            continue
        try:
            with open(saved, "r", encoding="utf-8") as f:
                d = json.load(f)
            sec = d.get("evaluation_settings", {}).get("patient_processing_time_sec")
            if sec is not None:
                rows.append({
                    "judge": r.get("judge_model"),
                    "orchestrator": r.get("orchestrator_model"),
                    "patient": r.get("patient_id"),
                    "sec": round(sec, 1),
                })
        except Exception as e:
            print(f"  Warning: could not read {saved}: {e}")

    if not rows:
        print("\nNo latency data found in results.")
        return

    import pandas as pd
    df = pd.DataFrame(rows)

    print("\n--- Per-judge × per-patient latency (seconds) ---")
    print(df.to_string(index=False))

    print("\n--- Per-judge mean latency ---")
    judge_means = df.groupby("judge")["sec"].mean().round(1).sort_values()
    for j, m in judge_means.items():
        print(f"  {j}: {m:.1f} s")

    print("\n--- Per-patient mean latency ---")
    pat_means = df.groupby("patient")["sec"].mean().round(1).sort_values()
    for p, m in pat_means.items():
        print(f"  {p}: {m:.1f} s")


def main() -> int:
    args = parse_args()

    # Load ground truth
    try:
        gt = load_gt(args.gt)
    except Exception as e:
        print(f"Failed to load ground truth: {e}")
        return 1

    # Resolve judge models
    if args.judges is None:
        judge_models = config.available_models  # all except bge/medgemma
    else:
        judge_models = args.judges

    # Run benchmark
    try:
        results = run_benchmark(
            data_dir=args.data_dir,
            gt_mapping=gt,
            judge_models=judge_models,
            max_files=args.max_files,
            no_restart=args.no_restart,
            orchestrators=args.orchestrators,
        )
    except Exception as e:
        print(f"\nBenchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Summarize latency
    summarize_latency(results)

    return 0


if __name__ == "__main__":
    sys.exit(main())