# Findings Report — Evaluation Session 2026-07-16

> **Cohort caveat:** this report describes results on a **small pilot cohort of
> 20 patients**. The figures are intended to validate the framework and
> demonstrate the analysis workflow, **not** to draw final clinical
> conclusions. The finalized study is planned on a cohort of ~300 patients.

## 1. Setup

* **Session:** `eval/2026-07-16/15-28-33` (anonymized copy in `results/session_2026-07-16/`)
* **Patients:** 20 (identifiers anonymized to `P01`–`P20`)
* **Models (open-weight, GGUF via `llama-cpp-server` / CUDA):**
  `baichuan-m2-32b`, `gpt-oss-20b`, `huatuoGPT-3-32b`,
  `nemotron-3-nano-30B-A3B`, `qwen3-30B-A3B-Thinking`
* **Design:** all-to-all evaluation matrix — every model acts as both
  **orchestrator** (generating the treatment plan) and **judge** (scoring every
  orchestrator, including itself).
* **Total evaluations:** 454

## 2. Aggregate Results

| Metric | Mean |
|---|---|
| Reasoning Efficiency Score (RES) | **65.9%** |
| Binary clinical accuracy (IO vs IOCT) | **0.678** |
| Reasoning Completeness (8-item checklist) | **80.7%** |
| Treatment Plan Completeness (8-item checklist) | **93.3%** |
| Factuality | **86.2%** |
| Faithfulness (Likert 1–5) | **3.49** |

The pipeline was highly stable: **0 zero-score cases** in this session.

## 3. Consensus Leaderboard (per orchestrator)

Consensus = average over all judges for continuous metrics; majority vote for
binary accuracy.

| Orchestrator | RES (mean ± std) | Binary accuracy |
|---|---|---|
| `nemotron-3-nano-30B-A3B` | 70.0 ± 12.3 | 0.550 |
| `huatuoGPT-3-32b` | 69.7 ± 22.8 | 0.692 |
| `baichuan-m2-32b` | 69.6 ± 10.6 | **0.800** |
| `gpt-oss-20b` | 63.1 ± 11.5 | 0.650 |
| `qwen3-30B-A3B-Thinking` | 58.5 ± 13.3 | 0.700 |

`baichuan-m2-32b` leads on binary accuracy while RES is led by
`nemotron-3-nano-30B-A3B` and `huatuoGPT-3-32b`.

## 4. Judge-level observations

| Judge | Mean RES | Binary accuracy |
|---|---|---|
| `baichuan-m2-32b` | 68.5 | 0.659 |
| `gpt-oss-20b` | 56.6 | 0.602 |
| `huatuoGPT-3-32b` | **76.5** | 0.667 |
| `nemotron-3-nano-30B-A3B` | 66.0 | 0.720 |
| `qwen3-30B-A3B-Thinking` | 62.5 | **0.742** |

### Key findings

1. **Systemic positive (IOCT) bias.** Across almost all configurations the
   models predict `IOCT [1]` more often than it occurs in the ground truth
   (predicted 301 positive vs 273 true positive; 153 predicted negative vs 181
   true negative). High sensitivity is traded for frequent false positives.
2. **Judge rigidity.** Advanced judges (`nemotron-3-nano-30B-A3B`,
   `qwen3-30B-A3B-Thinking`) produce near-identical evaluation matrices
   regardless of the orchestrator they score — strong internal criteria that
   overwrite inter-orchestrator differences.
3. **Self-evaluation is not a guaranteed advantage.** `baichuan-m2-32b` is the
   only model whose self-evaluation is notably balanced (self-eval binary
   accuracy 0.842, its best). Others (e.g. `gpt-oss-20b`, 0.550) do not
   benefit from judging themselves.
4. **Orchestrator stability.** `baichuan-m2-32b` most consistently preserves
   the `IO [0]` class (high true negatives) across judges.

## 5. Reproducing

```bash
# 1. Re-run an evaluation session on your own data
python -m nsclc_eval.cli --data-dir data/your_patients --gt data/your_ground_truth.csv

# 2. Analyze the outputs
# Open notebooks/post_run_analysis.ipynb and point EVAL_SESSION_PATH at the
# resulting eval/<date>/<time> directory.
```

All figures in this report can be regenerated from
`results/session_2026-07-16/` with `notebooks/post_run_analysis.ipynb`.
