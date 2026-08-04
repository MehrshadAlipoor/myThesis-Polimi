# Legacy prototype

This folder contains the **initial prototype** of the evaluation framework,
built on **Ollama** (`qwen3:14b`) before the project was rewritten into the
modular `src/nsclc_eval/` package.

* `models.py` — Pydantic models for reasoning decomposition (MedR-Bench style)
  and the NOHARM safety report.
* `prompts.py` — The two original prompt templates (atomic step parsing +
  safety/effectiveness evaluation).
* `evaluator.py` — Two-phase evaluation: Reasoning Efficiency + Safety Index.

This code is kept for reference / reproducibility of the early experiments and
is **not** part of the active pipeline.
