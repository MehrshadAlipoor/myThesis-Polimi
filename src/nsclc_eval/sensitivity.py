"""Prompt-sensitivity testing (PSS/JSS) for the judge prompts.

Three semantically equivalent **variant sets** of the three critical judge
prompts (``v0`` = canonical baseline, ``v1``/``v2`` = paraphrases preserving
intent, output schema and enum vocabulary). Each sensitivity run overrides all
three prompts together, so every metric stays available while wording varies.

Speed edition: v1/v2 reuse v0's **prompt-invariant** metric results
(completeness, GAR, treatment completeness, output factuality, binary accuracy),
whose prompts are not part of the variant sets. For those metrics PSS=0 /
JSS=1 by construction and should be reported as "prompt-invariant".
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import config
from .prompts import (
    BATCHED_CLASSIFICATION_PROMPT,
    CONCISE_NOTE,
    FACTUALITY_PROMPT,
    FAITHFULNESS_PROMPT,
)
from .pipeline import run_evaluation
from .utils import _sanitize_filename, build_file_index


# =====================================================================
# Override context managers
# =====================================================================

@contextlib.contextmanager
def use_prompt_overrides(overrides: Dict[str, str]):
    """Temporarily override prompt constants for one run (sequential per variant)."""
    import nsclc_eval.prompts as prompts_mod
    prev = dict(prompts_mod.ACTIVE_PROMPTS)
    prompts_mod.ACTIVE_PROMPTS.update(overrides)
    try:
        yield
    finally:
        prompts_mod.ACTIVE_PROMPTS = prev


@contextlib.contextmanager
def use_output_subdir(subdir: str):
    """Route ``save_evaluation_results`` into ``eval/.../<judge>/<subdir>/``."""
    import nsclc_eval.utils as utils_mod
    prev = utils_mod.OUTPUT_SUBDIR
    utils_mod.OUTPUT_SUBDIR = subdir
    try:
        yield
    finally:
        utils_mod.OUTPUT_SUBDIR = prev


# =====================================================================
# Prompt variant sets (v0 = canonical, v1/v2 = paraphrases)
# =====================================================================

V0 = {
    "BATCHED_CLASSIFICATION_PROMPT": BATCHED_CLASSIFICATION_PROMPT,
    "FACTUALITY_PROMPT": FACTUALITY_PROMPT,
    "FAITHFULNESS_PROMPT": FAITHFULNESS_PROMPT,
}

CLASS_V1 = """
# Task Description
You will receive a numbered list of reasoning steps produced by a clinical AI agent that selects therapy for NSCLC.
Categorize EVERY step into exactly one of:
1. Citation: Restatement of record info without new reasoning.
2. Repetition: Repetition of previous steps without advancing the process.
3. Reasoning: Deriving new conclusions that move toward the correct answer.
4. Redundancy: New info that does not help reach the final answer.

# Note
Consider each step relative to ALL other steps, the patient's medical record, and the final reasoning goal. If a step matches multiple types, pick the one that best reflects its contribution to the reasoning goal. Maintain objectivity; avoid subjective assumptions.

# Output Format (JSON)
{"assessments": [{"step_id": 1, "classification": "Citation", "rationale": "short reason"}, ...]}

Rules:
- Return ONE entry per input step, using the SAME step numbers as step_id.
- "classification" must be exactly one of "Citation", "Repetition", "Reasoning", "Redundancy".
- Keep each rationale under 25 words.

Here is an example classification set for a fictional case:
{"assessments": [
 {"step_id": 1, "classification": "Citation", "rationale": "Restates presenting symptoms"},
 {"step_id": 2, "classification": "Reasoning", "rationale": "Derives meningitis hypothesis"},
 {"step_id": 3, "classification": "Repetition", "rationale": "Repeats headache fact"},
 {"step_id": 4, "classification": "Redundancy", "rationale": "Irrelevant unverified claim"},
 {"step_id": 5, "classification": "Redundancy", "rationale": "Eye color is irrelevant"}
]}
"""

CLASS_V2 = """
# Task
Below is a numbered list of reasoning steps from an NSCLC therapy-selection AI agent.
Assign a single label to EVERY step, choosing one of:
- "Citation"    (re-states record information; adds no new reasoning)
- "Repetition"  (repeats an earlier step; does not advance)
- "Reasoning"   (draws a new inference that moves toward the answer)
- "Redundancy"  (new but unhelpful information for reaching the answer)

# Guidance
Judge each step in the context of the other steps, the medical record, and the final goal. For ambiguous steps, choose the label that best captures its role. Stay objective.

# Output (JSON only)
{"assessments": [{"step_id": 1, "classification": "Citation", "rationale": "..."}, ...]}
- One object per step; keep the input step numbers.
- "classification" must be one of "Citation", "Repetition", "Reasoning", "Redundancy".
- Rationale under 25 words.

Example:
{"assessments": [
 {"step_id": 1, "classification": "Citation", "rationale": "Restates symptoms"},
 {"step_id": 2, "classification": "Reasoning", "rationale": "Infers diagnosis"},
 {"step_id": 3, "classification": "Repetition", "rationale": "Repeats symptom"},
 {"step_id": 4, "classification": "Redundancy", "rationale": "Irrelevant claim"},
 {"step_id": 5, "classification": "Redundancy", "rationale": "Eye color is irrelevant"}
]}
"""

FACT_V1 = """
# Task Description
You are a medical claim verifier for NSCLC therapy AI agents.

You will receive:
1. **PATIENT_DATA**: The actual clinical record fields of a patient (structured key-value pairs).
2. **REASONING_STEPS**: Numbered reasoning steps produced by an AI agent.

# Your Task
- Scan the reasoning steps and identify every **factual claim** the agent makes about the patient's clinical data.
- For each claim, check whether it matches PATIENT_DATA.
- Label each claim's verdict as: "correct", "incorrect", or "unsupported".
- Compute `factuality_score` as `(correct_claims / total_claims) * 100`. If no claims found, set to 100.0.

# Important Rules
- Only extract claims about patient-specific clinical facts, NOT general medical knowledge.
- If a field has multiple values or aliases (e.g., "WT" and "Wild-type"), treat them as equivalent.

# Output Format (JSON)
Each claim MUST have exactly these 5 fields:
- "step_number": integer (which reasoning step this claim comes from)
- "claim": string (the factual claim text)
- "referenced_field": string (which clinical field, e.g. "histology", "pdl1_tps")
- "actual_value": string (the actual value from PATIENT_DATA)
- "verdict": string (one of: "correct", "incorrect", "unsupported")

Example output:
{"claims": [{"step_number": 1, "claim": "Patient has adenocarcinoma", "referenced_field": "histology", "actual_value": "Adenocarcinoma", "verdict": "correct"}], "total_claims": 1, "correct_claims": 1, "incorrect_claims": 0, "unsupported_claims": 0, "factuality_score": 100.0}
"""

FACT_V2 = """
# Role
Act as a clinical fact-checker that verifies claims made by NSCLC therapy AI agents.

# Inputs
1. PATIENT_DATA: the patient's actual structured clinical record.
2. REASONING_STEPS: numbered reasoning steps from an AI agent.

# Steps
- List every factual claim the agent asserts about patient-specific clinical data.
- Compare each claim against PATIENT_DATA.
- Verdict each claim "correct", "incorrect", or "unsupported".
- factuality_score = correct_claims / total_claims * 100 (use 100.0 if no claims).

# Rules
- Ignore general medical statements; only patient-specific facts count.
- Treat equivalent aliases as equal (e.g. "WT" == "Wild-type").

# JSON Output — five fields per claim
{"claims": [{"step_number": 1, "verdict": "correct", "claim": "Patient has adenocarcinoma", "referenced_field": "histology", "actual_value": "Adenocarcinoma"}],
 "total_claims": 1, "correct_claims": 1, "incorrect_claims": 0, "unsupported_claims": 0, "factuality_score": 100.0}
"""

FAITH_V1 = """
# Task Description
You are a logical coherence evaluator for NSCLC therapy AI agents.

You will receive:
1. **REASONING_STEPS**: Numbered reasoning steps from an AI agent for NSCLC therapy selection.
2. **FINAL_RECOMMENDATION**: The agent's final therapy recommendation text.

# Your Task
1. **Self-contradictions**: Check if any step contradicts another step.
2. **Conclusion follows**: Check if the final recommendation logically follows from the reasoning steps.
3. **Logical gaps**: Identify steps that make claims without supporting reasoning or skip important logical links.

# Rating Scale (faithfulness_score)
- 5 = fully coherent, no contradictions, conclusion follows
- 4 = minor gaps, no contradictions, conclusion mostly follows
- 3 = one contradiction or one major gap
- 2 = several contradictions or conclusion unsupported
- 1 = severely incoherent, many contradictions

# Output Format (JSON)
- "contradictions": list of {"step_a": int, "step_b": int, "description": string}
- "conclusion_follows": boolean
- "logical_gaps": list of {"step_number": int, "description": string}
- "faithfulness_score": integer (1-5)
- "summary": string
"""

FAITH_V2 = """
# Role
Assess the internal logical soundness of an NSCLC therapy AI agent's reasoning chain.

# Inputs
1. REASONING_STEPS: numbered steps from the agent.
2. FINAL_RECOMMENDATION: the agent's final recommendation.

# Checks
- Detect contradictions between steps (report {"step_a", "step_b", "description"}).
- Judge whether the recommendation follows from the steps ("conclusion_follows").
- List logical gaps ("logical_gaps": [{"step_number", "description"}]).

# Score (faithfulness_score, 1-5)
5 = perfectly coherent; 4 = minor gaps only; 3 = one contradiction/gap;
2 = multiple contradictions or unsupported conclusion; 1 = incoherent.

# JSON Output
{"contradictions": [], "conclusion_follows": true, "logical_gaps": [],
 "faithfulness_score": 5, "summary": "..."}
"""

PROMPT_VARIANT_SETS = [
    V0,
    {
        "BATCHED_CLASSIFICATION_PROMPT": CLASS_V1,
        "FACTUALITY_PROMPT": FACT_V1 + CONCISE_NOTE,
        "FAITHFULNESS_PROMPT": FAITH_V1 + CONCISE_NOTE,
    },
    {
        "BATCHED_CLASSIFICATION_PROMPT": CLASS_V2,
        "FACTUALITY_PROMPT": FACT_V2 + CONCISE_NOTE,
        "FAITHFULNESS_PROMPT": FAITH_V2 + CONCISE_NOTE,
    },
]


# =====================================================================
# Sensitivity runner
# =====================================================================

SENSITIVITY_JUDGES = config.available_models
SENSITIVITY_ORCHESTRATOR = "baichuan-m2-32b"
SENSITIVITY_PATIENTS = 20

# Reuse v0's prompt-invariant metrics in v1/v2 (saves ~40% of the run).
SENSITIVITY_REUSE_INVARIANT = True


def _fingerprint(variant_set: Dict[str, str]) -> Dict[str, str]:
    return {k: hashlib.sha256(v.encode("utf-8")).hexdigest()[:8]
            for k, v in variant_set.items()}


def _load_cached_invariant_metrics(run_date: str, run_timestamp: str, judge: str,
                                   orchestrator: str, patient_ids: List[str]) -> Dict[str, Dict]:
    """Load prompt-invariant metric dicts from v0's saved per-patient JSONs."""
    base = f"eval/{run_date}/{run_timestamp}/{_sanitize_filename(judge)}/sensitivity/v0"
    cache = {}
    for pid in patient_ids:
        path = f"{base}/{pid}_evaluation_{_sanitize_filename(orchestrator)}.json"
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        cache[pid] = {
            "completeness": d.get("metrics_tier1_completeness"),
            "gar": d.get("metrics_tier1_guideline_adherence"),
            "treat_comp": d.get("metrics_tier2_treatment_completeness"),
            "out_fact": d.get("metrics_tier2_factuality"),
            "tier2": d.get("metrics_tier2"),
        }
    return cache


def run_sensitivity_experiment(
    data_dir: str,
    gt_mapping: Dict,
    judges: List[str],
    orchestrator: str,
    prompt_variant_sets: Optional[List[Dict]] = None,
    max_patients: Optional[int] = None,
    mode: str = "both",
    eval_flags: Optional[Dict] = None,
    save_results: bool = True,
    verbose: bool = True,
    restart_server: bool = True,
):
    from datetime import datetime
    prompt_variant_sets = prompt_variant_sets or PROMPT_VARIANT_SETS

    orch_to_files, _ = build_file_index(data_dir)
    files = orch_to_files.get(orchestrator, [])
    if not files:
        raise RuntimeError(f"No files found for orchestrator '{orchestrator}'")
    if max_patients is not None:
        files = files[:max_patients]

    run_date = datetime.now().strftime("%Y-%m-%d")
    run_timestamp = datetime.now().strftime("%H-%M-%S")
    records = []
    current_judge = None

    for judge in judges:
        if restart_server and judge != current_judge:
            print(f"\n🔄 Swapping judge → {judge}")
            config.restart_llama_server(judge)
        current_judge = judge

        invariant_cache = None
        for vi, variant_set in enumerate(prompt_variant_sets):
            tag = f"v{vi}"
            fp = _fingerprint(variant_set)
            reuse = SENSITIVITY_REUSE_INVARIANT and tag != "v0" and invariant_cache is not None
            print(f"\n=== {judge} | {tag} | fingerprints={fp} | reuse_invariant={reuse} ===")
            with use_prompt_overrides(variant_set):
                with use_output_subdir(f"sensitivity/{tag}"):
                    results = run_evaluation(
                        file_paths=files, gt_mapping=gt_mapping, mode=mode,
                        judge_model=judge, orchestrator_model=orchestrator,
                        save_results=save_results, verbose=verbose,
                        run_date=run_date, run_timestamp=run_timestamp,
                        eval_flags=eval_flags,
                        cached_metrics_by_patient=(invariant_cache if reuse else None),
                    )
            if tag == "v0" and SENSITIVITY_REUSE_INVARIANT:
                patient_ids = [r["patient_id"] for r in results if r]
                invariant_cache = _load_cached_invariant_metrics(
                    run_date, run_timestamp, judge, orchestrator, patient_ids)
                print(f"  Cached prompt-invariant metrics for {len(invariant_cache)} "
                      f"patients (reused in v1/v2).")
            for r in results:
                r["variant"] = tag
                r["prompt_fingerprints"] = json.dumps(fp, sort_keys=True)
                records.append(r)

    df = pd.DataFrame(records)
    out_dir = f"eval/{run_date}/{run_timestamp}/sensitivity"
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(f"{out_dir}/sensitivity_records.csv", index=False)
    print(f"\n✅ Sensitivity records → {out_dir}/sensitivity_records.csv ({len(df)} rows)")
    return df, run_date, run_timestamp


# =====================================================================
# PSS & JSS computation
# =====================================================================

METRIC_LIST = [
    "res_score", "factuality_score", "faithfulness_score",
    "reasoning_completeness_score", "treatment_completeness_score",
    "gar_score", "output_factuality_score", "binary_accuracy", "predicted_label",
]

CONTINUOUS_METRICS = {
    "res_score", "factuality_score", "reasoning_completeness_score",
    "treatment_completeness_score", "gar_score", "output_factuality_score",
}
DISCRETE_METRICS = {"binary_accuracy", "predicted_label", "faithfulness_score"}


def compute_sensitivity_metrics(df: pd.DataFrame, metric_cols: Optional[List[str]] = None,
                                cont_tol: float = 1.0) -> pd.DataFrame:
    """PSS (per-patient spread) and JSS (fraction of identical scores)."""
    metric_cols = metric_cols or METRIC_LIST
    out = []
    for judge in df["judge_model"].unique():
        jdf = df[df["judge_model"] == judge]
        for metric in metric_cols:
            if metric not in jdf.columns:
                continue
            piv = jdf.pivot_table(index="patient_id", columns="variant", values=metric)
            piv = piv.dropna()
            if piv.empty or piv.shape[1] < 2:
                continue
            if metric in DISCRETE_METRICS:
                pss = np.nan
                cv = np.nan
                jss = (piv.nunique(axis=1) == 1).mean()
            else:
                pss = piv.std(axis=1, ddof=1).mean()
                pm = piv.mean(axis=1)
                cv = (piv.std(axis=1, ddof=1) / pm.replace(0, np.nan) * 100).mean()
                jss = ((piv.max(axis=1) - piv.min(axis=1)) <= cont_tol).mean()
            out.append({"judge_model": judge, "metric": metric,
                        "mean_pss": pss, "mean_cv_pct": cv, "jss": jss,
                        "n": int(piv.shape[0])})
    return pd.DataFrame(out)


def plot_sensitivity_boxplots(df: pd.DataFrame, metric_cols: Optional[List[str]] = None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    for metric in (metric_cols or METRIC_LIST):
        plt.figure(figsize=(9, 4))
        sns.boxplot(data=df, x="variant", y=metric, hue="judge_model")
        plt.title(metric)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


def plot_jss_heatmap(sens_metrics: pd.DataFrame):
    import matplotlib.pyplot as plt
    import seaborn as sns
    piv = sens_metrics.pivot(index="judge_model", columns="metric", values="jss")
    plt.figure(figsize=(10, 5))
    sns.heatmap(piv.astype(float), annot=True, fmt=".2f", cmap="viridis", vmin=0, vmax=1)
    plt.title("JSS — fraction of patients with identical score across variants")
    plt.tight_layout()
    plt.show()


def plot_cv_heatmap(sens_metrics: pd.DataFrame):
    import matplotlib.pyplot as plt
    import seaborn as sns
    piv = sens_metrics.pivot(index="judge_model", columns="metric", values="mean_cv_pct")
    plt.figure(figsize=(10, 5))
    sns.heatmap(piv.astype(float), annot=True, fmt=".1f", cmap="rocket_r")
    plt.title("CV% — relative spread across variants (lower = less prompt-sensitive)")
    plt.tight_layout()
    plt.show()
