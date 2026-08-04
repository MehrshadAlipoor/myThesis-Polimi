"""Command-line interface for the NSCLC evaluation framework.

Examples
--------
Run a smoke test on the bundled synthetic data::

    python -m nsclc_eval.cli --data-dir data/sample --gt data/sample/sample_ground_truth.csv \
        --judges 2 --orchestrators 1 --max-files 2

Run a full evaluation matrix::

    python -m nsclc_eval.cli --data-dir data/sample --gt data/sample/sample_ground_truth.csv
"""

from __future__ import annotations

import argparse

from . import config
from .data_loading import load_ground_truth
from .pipeline import DEFAULT_EVAL_FLAGS, run_all_patient_evaluations


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nsclc_eval",
        description="LLM-as-a-Judge evaluation of NSCLC therapy agents.",
    )
    parser.add_argument("--data-dir", default=config.DATA_DIR,
                        help="Directory of patient JSON files.")
    parser.add_argument("--gt", default=config.GROUND_TRUTH_CSV,
                        help="Ground-truth CSV path.")
    parser.add_argument(
        "--models-ini",
        default=config.MODELS_INI,
        help="Path to the models.ini registry.",
    )
    parser.add_argument(
        "--judges",
        type=int,
        default=None,
        help="Number of judge models to use (default: all from models.ini).",
    )
    parser.add_argument(
        "--orchestrators",
        nargs="+",
        default=None,
        help="Explicit orchestrator model names (default: all discovered from data dir).",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum patient files per batch (smoke-testing).",
    )
    parser.add_argument(
        "--mode",
        default="both",
        choices=["both", "tier1", "tier2"],
        help="Which tiers to evaluate.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not persist evaluation results to disk.",
    )
    parser.add_argument(
        "--no-restart",
        action="store_true",
        help="Do not restart the Docker llama-cpp-server between judges.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    config.MODELS_INI = args.models_ini
    config.GROUND_TRUTH_CSV = args.gt
    config.DATA_DIR = args.data_dir
    config._registry.read(args.models_ini)
    config.available_models = [
        m for m in config._registry.sections() if "bge" not in m.lower()]
    config.JUDGE_MODELS = config.available_models
    config.JUDGE_MODEL = config.available_models[0] if config.available_models else "qwen2.5:7b"

    judge_models = config.JUDGE_MODELS[:
                                       args.judges] if args.judges else config.JUDGE_MODELS

    print(f"Data dir : {args.data_dir}")
    print(f"GT csv   : {args.gt}")
    print(f"Judges   : {judge_models}")

    gt_mapping = load_ground_truth(args.gt)
    print(f"Ground truth loaded: {len(gt_mapping)} patients")

    results = run_all_patient_evaluations(
        data_dir=args.data_dir,
        gt_mapping=gt_mapping,
        judge_models=judge_models,
        mode=args.mode,
        orchestrators_to_evaluate=args.orchestrators,
        max_files_per_batch=args.max_files,
        save_results=not args.no_save,
        verbose=True,
        eval_flags=DEFAULT_EVAL_FLAGS,
        restart_server=not args.no_restart,
    )
    print(f"\nTotal evaluations completed: {len(results)}")


if __name__ == "__main__":
    main()
