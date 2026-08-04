"""Offline end-to-end smoke test for the modularized framework.

Validates that the full pipeline (data loading -> Tier 1 Core -> Tier 1
Extended -> Tier 2 -> persistence) runs correctly using a deterministic mock
LLM client, so it works without a live llama-cpp-server.

Run from the repo root:

    python scripts/smoke_test.py
"""

from nsclc_eval.testing import MockLLMClient
from nsclc_eval.pipeline import run_all_patient_evaluations
from nsclc_eval.data_loading import load_ground_truth
from nsclc_eval import config
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

# The package logs contain emoji (✅/⚠️) which the default Windows console
# codec (cp1252) cannot encode. Force UTF-8 for this script.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    config.configure(
        client_instance=MockLLMClient(),
        judge_models=["mock-judge-1", "mock-judge-2"],
        temperature=0.2,
        seed=42,
    )

    gt = load_ground_truth(
        str(REPO_ROOT / "data/sample/sample_ground_truth.csv"))
    print(f"Ground truth loaded: {len(gt)} patients")

    results = run_all_patient_evaluations(
        data_dir=str(REPO_ROOT / "data/sample"),
        gt_mapping=gt,
        judge_models=config.JUDGE_MODELS,
        orchestrators_to_evaluate=None,
        max_files_per_batch=3,
        save_results=True,
        verbose=True,
        restart_server=False,
    )

    print("\n" + "=" * 70)
    print(f"SMOKE TEST COMPLETED: {len(results)} evaluations")
    print("=" * 70)
    for r in results:
        print(
            f"  {r['patient_id']}: RES={r['res_score']} Acc={r['binary_accuracy']} "
            f"GAR={r['gar_score']} RC={r['reasoning_completeness_score']} "
            f"TC={r['treatment_completeness_score']} Fact={r['factuality_score']} "
            f"Faith={r['faithfulness_score']}"
        )

    assert len(results) > 0, "Expected at least one evaluation result"
    assert all(r["res_score"] ==
               100.0 for r in results), "Mock RES should be 100"
    print("\n✅ Smoke test passed.")


if __name__ == "__main__":
    main()
