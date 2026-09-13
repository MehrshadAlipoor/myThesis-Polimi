"""Unit tests for prompt-sensitivity metrics and the thinking-budget helper."""

import pandas as pd

from nsclc_eval import config
from nsclc_eval.sensitivity import compute_sensitivity_metrics


def test_thinking_budget_toggle():
    # Default: thinking judges get a cap, others do not.
    assert config._thinking_budget_for("baichuan-m2-32b") == 1024
    assert config._thinking_budget_for("gpt-oss-20b") is None

    prev = config.THINKING_BUDGET_ENABLED
    try:
        config.THINKING_BUDGET_ENABLED = False
        assert config._thinking_budget_for("baichuan-m2-32b") is None
    finally:
        config.THINKING_BUDGET_ENABLED = prev


def test_pss_jss_identical_scores_are_insensitive():
    rows = []
    for pid in ("P1", "P2", "P3"):
        for variant, score in (("v0", 60.0), ("v1", 60.0), ("v2", 60.0)):
            rows.append({"judge_model": "J", "patient_id": pid,
                         "variant": variant, "res_score": score,
                         "binary_accuracy": 1})
    df = pd.DataFrame(rows)
    metrics = compute_sensitivity_metrics(df, metric_cols=["res_score", "binary_accuracy"])
    res = metrics[metrics["metric"] == "res_score"].iloc[0]
    assert res["mean_pss"] == 0.0
    assert res["mean_cv_pct"] == 0.0
    assert res["jss"] == 1.0
    assert res["n"] == 3
    bin_row = metrics[metrics["metric"] == "binary_accuracy"].iloc[0]
    assert bin_row["jss"] == 1.0


def test_pss_detects_variation_and_jss_tolerance():
    rows = []
    for pid in ("P1", "P2"):
        for variant, score in (("v0", 50.0), ("v1", 70.0), ("v2", 50.0)):
            rows.append({"judge_model": "J", "patient_id": pid,
                         "variant": variant, "res_score": score})
    df = pd.DataFrame(rows)
    metrics = compute_sensitivity_metrics(df, metric_cols=["res_score"])
    row = metrics.iloc[0]
    # Spread of 20 on dis/agreeing triplets -> non-zero PSS and JSS < 1.
    assert row["mean_pss"] > 0
    assert row["jss"] == 0.0
