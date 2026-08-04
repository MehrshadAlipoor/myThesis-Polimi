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
│      ├─ TIER 1 CORE   : decompose → classify steps → RES             │
│      └─ (parallel) TIER 1 EXTENDED + TIER 2:                         │
│          ├─ Reasoning Completeness      (LLM, 8-item checklist)      │
│          ├─ Guideline Adherence (GAR)   (LLM, ESMO/ASCO guidelines)  │
│          ├─ Factuality                  (LLM, claim verification)    │
│          ├─ Faithfulness                (LLM, Likert 1-5)            │
│          ├─ Treatment Plan Completeness (LLM, 8-item checklist)      │
│          └─ Binary Clinical Accuracy    (LLM, IO vs IOCT)            │
└──────────────────────────────────────────────────────────────────────┘
```

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

Structured LLM outputs are parsed with **Pydantic** models whose
`model_validator(mode='before')` hooks remap the alternative field names that
small open-weight LLMs frequently return — making the pipeline robust to
schema hallucinations.

## Repository Layout

```
.
├── src/nsclc_eval/            # The evaluation package
│   ├── config.py              # Client, model registry, Docker VRAM management
│   ├── models.py              # Pydantic models (with field-name remapping)
│   ├── prompts.py             # All 8 prompt templates
│   ├── data_loading.py        # Patient JSON parsing + ground truth
│   ├── tier1_core.py          # RES decomposition & classification
│   ├── tier1_extended.py      # Completeness, Factuality, Faithfulness, GAR
│   ├── tier2_clinical.py      # Binary accuracy + treatment completeness
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
git clone https://github.com/MehrshadAlipoor/myThesis.git
cd myThesis
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

| Metric | Mean |
|---|---|
| RES | 65.9% |
| Binary accuracy | 0.678 |
| Reasoning Completeness | 80.7% |
| Treatment Completeness | 93.3% |
| Factuality | 86.2% |
| Faithfulness | 3.49 / 5 |

Key observations: a **systemic positive (IOCT) bias** across models, **judge
rigidity** in the strongest judges, and notably **balanced self-evaluation**
for `baichuan-m2-32b` (self-eval accuracy 0.842).

## License

MIT — see `LICENSE`.
