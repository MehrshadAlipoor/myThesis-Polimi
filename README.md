<div align="center">

# nsclc-eval

**A multi-tier LLM-as-a-Judge evaluation framework for AI agents that recommend first-line NSCLC therapy**

Master's thesis project — Politecnico di Milano, 2026

</div>

---

## Overview

`nsclc-eval` evaluates AI agents that generate **first-line treatment
recommendations for non-small-cell lung cancer (NSCLC)** — specifically the
binary choice between *immunotherapy alone* (IO) and *immunotherapy +
chemotherapy* (IOCT).

The framework uses **open-weight LLMs** as both the *orchestrator* (the agent
under test, producing treatment plans from patient data) and the *judge*
(scoring those plans). Evaluation runs in an **all-to-all matrix**: every model
judges every other model's output, including its own, exposing judge bias,
rigidity, and cross-model consistency.

```
┌──────────────────────────────────────────────────────────────────────┐
│  run_all_patient_evaluations()                                        │
│  ├─ for each judge × orchestrator pair:                              │
│  │   └─ restart llama-cpp-server Docker (VRAM flush on judge swap)   │
│  └─ run_evaluation() for each patient file:                          │
│      ├─ TIER 1 CORE   : resolve steps → classify (1 batched call) →  │
│      │                  RES                                          │
│      └─ (parallel) TIER 1 EXTENDED + TIER 2:                         │
│          ├─ Reasoning Completeness      (LLM, 8-item checklist)      │
│          ├─ Guideline Adherence (GAR)   (LLM, ESMO/ASCO guidelines)  │
│          ├─ Factuality                  (LLM, claim verification)    │
│          ├─ Faithfulness                (LLM, Likert 1-5)            │
│          ├─ Treatment Plan Completeness (LLM, 8-item checklist)      │
│          ├─ Output Factuality           (LLM, claim verification)    │
│          └─ Binary Clinical Accuracy    (LLM, IO vs IOCT)            │
└──────────────────────────────────────────────────────────────────────┘
```

**Speed edition.** Judge generation is bounded three ways: per-metric
`max_tokens` caps, a per-request `thinking_budget_tokens` cap for the
thinking judges (CoT stays ON, just bounded), and a pinned server config
(`-ngl 999`, `--ubatch-size 512`, KV cache `q8_0`, `--cache-prompt`). The
sensitivity runner additionally reuses v0's prompt-invariant metrics in v1/v2.

**Prompt sensitivity.** `nsclc_eval.sensitivity` runs the three critical judge
prompts (classification, factuality, faithfulness) under three semantically
equivalent variants and reports **PSS** (per-patient score spread) and **JSS**
(agreement across variants).

## Evaluation Tiers

| Tier | Metric | Type | Description |
|------|--------|------|-------------|
| **1 Core** | Reasoning Efficiency Score (RES) | LLM | Decomposes reasoning into atomic steps; classifies each as Citation / Repetition / Reasoning / Redundancy; `RES = reasoning / total × 100` |
| **1 Ext** | Reasoning Completeness | LLM | Checks whether 8 clinical fields (histology, PD-L1, ECOG, biomarkers, stage, …) are mentioned in the reasoning |
| **1 Ext** | Factuality | LLM | Verifies factual claims against the structured patient record → correct / incorrect / unsupported |
| **1 Ext** | Faithfulness | LLM | Rates internal consistency, contradictions and logical gaps (Likert 1–5) |
| **1 Ext** | Guideline Adherence (GAR) | LLM | Checks each reasoning step against retrieved ESMO/ASCO guidelines |
| **2** | Binary Clinical Accuracy | LLM | Predicts IO vs IOCT and compares with the clinical ground truth |
| **2** | Treatment Plan Completeness | LLM | Same 8-item checklist applied to the final treatment plan |
| **2** | Output Factuality | LLM | Claim verification over the agent's final output (not step-decomposed) |

Structured LLM outputs are parsed with **Pydantic** models whose
`model_validator(mode='before')` hooks remap the alternative field names that
small open-weight LLMs frequently return — making the pipeline robust to
schema hallucinations.

## Repository Layout

```
.
├── src/nsclc_eval/            # The evaluation package
│   ├── config.py              # Client, model registry, speed settings, Docker VRAM
│   ├── llm.py                 # Bounded-concurrency LLM calls + thinking budget
│   ├── models.py              # Pydantic models (with field-name remapping)
│   ├── prompts.py             # Prompt templates + sensitivity override hook
│   ├── data_loading.py        # Patient JSON parsing, ground truth, guideline GT
│   ├── tier1_core.py          # RES — batched step classification
│   ├── tier1_extended.py      # Completeness, Factuality, Faithfulness, GAR
│   ├── tier2_clinical.py      # Binary accuracy
│   ├── sensitivity.py         # Prompt variants, PSS/JSS, invariant caching
│   ├── utils.py               # File indexing & result persistence
│   ├── pipeline.py            # Orchestration (single / full matrix runs)
│   ├── cli.py                 # Command-line entry point
│   └── testing.py             # Deterministic mock LLM client (offline demos)
├── notebooks/
│   ├── 01_evaluation_demo.ipynb    # Offline end-to-end demo (mock client)
│   └── post_run_analysis.ipynb     # Unified analysis of a session's results
├── data/sample/               # SYNTHETIC demo patients + ground truth (no real data)
├── results/
│   ├── session_2026-07-16/    # Anonymized pilot-session summaries
│   └── findings_report.md     # Key findings from the pilot session
├── legacy/                    # Early Ollama-based prototype (reference only)
├── config/models.ini          # GGUF model registry for llama-cpp-server
├── scripts/smoke_test.py      # Offline end-to-end smoke test
└── tests/                     # Unit tests (mock LLM)
```

## Quick Start

### 1. Install

```bash
git clone https://github.com/AI-ON-Laboratory/2026_Alipoor_AgentEvalFramework.git
cd 2026_Alipoor_AgentEvalFramework
pip install -e .            # or: pip install -r requirements.txt
```

### 2. Offline demo (no GPU required)

The package ships with a **deterministic mock LLM client** so the full pipeline
can be exercised without a live server:

```bash
python scripts/smoke_test.py
# or open notebooks/01_evaluation_demo.ipynb and run all cells
```

### 3. Run against real open-weight models

The framework expects a local `llama-cpp-server` (CUDA) Docker container serving
the OpenAI-compatible API at `http://localhost:8080`. Model paths are declared
in `config/models.ini`.

```bash
python -m nsclc_eval.cli \
    --data-dir data/your_patients \
    --gt data/your_ground_truth.csv
```

Point `data-dir` at your patient JSON files and `--gt` at the semicolon-delimited
ground-truth CSV (columns `Subject;IO_IOCT`). Results are written to
`eval/<date>/<time>/<judge>/`.

### 4. Analyze a session

Open `notebooks/post_run_analysis.ipynb` and set `EVAL_SESSION_PATH` to the
`eval/<date>/<time>` directory. The notebook produces the pair-wise metric
matrix, judge-consistency bar charts, RES distributions, confusion matrices,
Precision/Recall/F1, ROC curves, radar charts and a consensus leaderboard.

### 5. Prompt sensitivity

Run the paraphrased-prompt experiment (PSS/JSS) directly from the CLI:

```bash
python -m nsclc_eval.cli --sensitivity \
    --data-dir data/your_patients --gt data/your_ground_truth.csv \
    --sensitivity-orchestrator baichuan-m2-32b --sensitivity-patients 20
```

Per-patient JSONs and summaries are written under
`eval/<date>/<time>/<judge>/sensitivity/v{k}/`, plus a flat
`sensitivity_records.csv`. v1/v2 reuse v0's prompt-invariant metrics
(completeness, GAR, treatment completeness, output factuality, binary accuracy);
pass `--no-reuse-invariant` to run the complete pipeline for every variant.

## Privacy & Data Handling

* The repository contains **no real patient data**.
* `data/sample/` holds **synthetic** patients fabricated for the demo.
* `results/session_2026-07-16/` contains **anonymized** aggregate summaries
  (patient IDs replaced with `P01`…`P20`, file paths redacted).
* The `.gitignore` excludes real data directories (`data/*`, `eval/`, `*.csv`,
  `*.xlsx`, model weights) so sensitive files can never be accidentally pushed.

## Pilot Results (toy cohort, n = 20)

Full details in [`results/findings_report.md`](results/findings_report.md).
These results validate the framework on a small pilot cohort and are **not**
final clinical conclusions — the finalized study targets ~300 patients.

**Design:** all-to-all evaluation matrix — the same 5 open-weight models act as
both *orchestrator* (generating the treatment plan) and *judge* (scoring every
orchestrator). This yields **25 judge × orchestrator pairs** and **454
evaluations** over 20 patients.

### 1. Aggregate across all judge × orchestrator pairs

Averages over the full matrix (all 25 pairs / 454 evaluations).

| Metric | Mean |
|---|---|
| Reasoning Efficiency Score (RES) | 65.9% |
| Binary clinical accuracy (IO vs IOCT) | 0.678 |
| Reasoning Completeness (8-item checklist) | 80.7% |
| Treatment Plan Completeness (8-item checklist) | 93.3% |
| Factuality | 86.2% |
| Faithfulness (Likert 1–5) | 3.49 / 5 |

### 2. Consensus per orchestrator (averaged over all judges)

Scores for each *orchestrator* (the agent under test), aggregated across all 5
judges (mean for continuous metrics, majority vote for accuracy).

| Orchestrator | RES (mean ± std) | Binary accuracy |
|---|---|---|
| `nemotron-3-nano-30B-A3B` | 70.0 ± 12.3 | 0.550 |
| `huatuoGPT-3-32b` | 69.7 ± 22.8 | 0.692 |
| `baichuan-m2-32b` | 69.6 ± 10.6 | **0.800** |
| `gpt-oss-20b` | 63.1 ± 11.5 | 0.650 |
| `qwen3-30B-A3B-Thinking` | 58.5 ± 13.3 | 0.700 |

### 3. Per judge (averaged over all orchestrators)

Scores of each *judge*, aggregated across all 5 orchestrators it evaluated.

| Judge | Mean RES | Binary accuracy |
|---|---|---|
| `baichuan-m2-32b` | 68.5 | 0.659 |
| `gpt-oss-20b` | 56.6 | 0.602 |
| `huatuoGPT-3-32b` | **76.5** | 0.667 |
| `nemotron-3-nano-30B-A3B` | 66.0 | 0.720 |
| `qwen3-30B-A3B-Thinking` | 62.5 | **0.742** |

### Key observations

* **Systemic positive (IOCT) bias** — models predict `IOCT [1]` more often than
  it occurs in the ground truth (301 predicted vs 273 true positive), trading
  high sensitivity for frequent false positives.
* **Judge rigidity** — the strongest judges (`nemotron-3-nano-30B-A3B`,
  `qwen3-30B-A3B-Thinking`) produce near-identical matrices regardless of the
  orchestrator they score.
* **Balanced self-evaluation** — `baichuan-m2-32b` is the only model whose
  self-evaluation is notably balanced (self-eval binary accuracy 0.842).

Per-pair breakdowns (all 25 combinations) are in
[`results/findings_report.md`](results/findings_report.md) and can be
regenerated from `notebooks/post_run_analysis.ipynb`.

## License

MIT — see `LICENSE`.
