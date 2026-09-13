"""Tier 1 Core — Reasoning Efficiency Score (RES).

Decomposes the raw agent reasoning into atomic steps, classifies each step as
*Citation / Repetition / Reasoning / Redundancy* and computes the
Reasoning Efficiency Score:

    RES = (reasoning_steps / total_steps) x 100

Speed edition: all steps are classified in a **single batched LLM call** (was
N parallel calls). When the upstream record already carries structured
``reasoning_steps``, the judge classifies those as-is (``steps_preserved``) and
LLM reformatting is only used as a fallback.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

from .llm import create_call, parse_call
from .models import (
    EfficiencyCategory,
    StepClassification,
    StepClassificationBatch,
)
from .prompts import P


def _truncate_degenerate_loop(text: str) -> str:
    """Cut pathological repetition loops produced by some judges."""
    if not text:
        return text
    m = re.search(r'(?:<Step>\s*>\s*){3,}', text, re.IGNORECASE)
    if m:
        return text[:m.start()].rstrip()
    m = re.search(r'((<[^>]{1,20}>)\s*){25,}', text)
    if m:
        return text[:m.start()].rstrip()
    return text


def _strip_think_blocks(text: str) -> str:
    """Remove reasoning-model thinking wrappers (<think>...</think>)."""
    if not text:
        return text
    text = re.sub(r'(?is)<\s*(?:think|thinking)\s*>.*?(?:<\s*/\s*(?:think|thinking)\s*>|\Z)', '', text)
    return text.strip()


def decompose_reasoning(raw_reasoning: str) -> str:
    """Ask the judge LLM to reorganize raw reasoning into atomic steps."""
    from . import config
    response = create_call(
        model=config.JUDGE_MODEL,
        messages=[
            {"role": "system", "content": P("REFORMAT_PROMPT")},
            {"role": "user", "content": f"Final report: {raw_reasoning}"},
        ],
        temperature=config.TEMPERATURE, seed=config.SEED,
        max_tokens=config.MAX_TOKENS["decompose"],
    )
    msg = response.choices[0].message
    content = (msg.content or "").strip()
    if not content:
        content = getattr(msg, "reasoning_content", None) or ""
    atomic_steps = _truncate_degenerate_loop(_strip_think_blocks(content))
    print("\n=== Atomic Steps from Rationale ===\n")
    print(atomic_steps)
    return atomic_steps


def _extract_step_texts(step_blob: str) -> List[str]:
    """Parse the LLM's reformatted output into a list of step texts."""
    if not step_blob:
        return []
    step_blob = _truncate_degenerate_loop(step_blob)
    step_header_pattern = re.compile(
        r'^(?:[-\*\s\u2022]*)?(?:<|\[)?Step\s*\d+\s*(?:>|\]|:|\)|\.)? *(.*)$',
        re.IGNORECASE,
    )
    numeric_pattern = re.compile(r'^(?:[-\*\s\u2022]*)\d+[\s\)\.:\-]+\s*(.*)$', re.IGNORECASE)
    extracted_steps = []
    current_step_lines = []
    lines = [line.strip() for line in str(step_blob).replace('\r', '\n').split('\n') if line.strip()]
    has_explicit_steps = any(step_header_pattern.match(l) or numeric_pattern.match(l) for l in lines)
    if has_explicit_steps:
        for line in lines:
            match = step_header_pattern.match(line) or numeric_pattern.match(line)
            if match:
                if current_step_lines:
                    extracted_steps.append(' '.join(current_step_lines).strip())
                    current_step_lines = []
                remainder = match.group(1).strip()
                if remainder:
                    current_step_lines.append(remainder)
            else:
                if current_step_lines:
                    current_step_lines.append(line)
        if current_step_lines:
            extracted_steps.append(' '.join(current_step_lines).strip())
    else:
        for line in lines:
            for seg in re.split(r'\s*[|;]\s*', line):
                clean_line = re.sub(r'^[-\*\u2022\s>+]+', '', seg).strip()
                if clean_line:
                    extracted_steps.append(clean_line)
    cleaned = []
    for step in extracted_steps:
        s = re.sub(r'^<+/?Step\s*>+$', '', step, flags=re.IGNORECASE).strip()
        if not s:
            continue
        cleaned.append(s[:600])
    return cleaned[:10]


def _resolve_steps(reasoning_steps: Optional[List[str]], raw_reasoning: str) -> Tuple[List[str], Optional[str], str]:
    """Return ``(steps, atomic_steps, status)`` for RES and Tier-1 metrics.

    Prefers the orchestrator's own structured ``reasoning_steps`` list so the
    judge classifies — never re-segments — the agent's reasoning; LLM
    reformatting is only a fallback for records that carry raw text.
    """
    if reasoning_steps:
        steps = [str(s).strip()[:600] for s in reasoning_steps if str(s).strip()][:20]
        print(f"  Using {len(steps)} preserved reasoning_steps (skipped LLM decompose).")
        return steps, None, "steps_preserved"
    atomic_steps = decompose_reasoning(raw_reasoning)
    steps = _extract_step_texts(atomic_steps)
    if steps:
        return steps, atomic_steps, "llm_decomposed"
    print("  ⚠️ Decompose returned no steps; falling back to raw reasoning text.")
    return _extract_step_texts(raw_reasoning)[:10], atomic_steps, "fallback_raw"


def classify_reasoning_steps(atomic_steps: Optional[str], steps: Optional[List[str]] = None) -> List[StepClassification]:
    """One batched call classifies all steps; results printed sequentially."""
    from . import config
    if steps is None:
        steps = _extract_step_texts(atomic_steps)
    print(f"\nFound {len(steps)} steps to assess (single batched call).\n")
    if not steps:
        return []

    steps_text = "\n".join(f"<Step {i}> {t}" for i, t in enumerate(steps, 1))
    try:
        response = parse_call(
            model=config.JUDGE_MODEL,
            messages=[
                {"role": "system", "content": P("BATCHED_CLASSIFICATION_PROMPT")},
                {"role": "user", "content": f"{steps_text}\n\nTotal number of steps: {len(steps)}"},
            ],
            response_format=StepClassificationBatch,
            temperature=config.TEMPERATURE, seed=config.SEED,
            max_tokens=config.MAX_TOKENS["classify"],
        )
        parsed = response.choices[0].message.parsed
    except Exception as e:
        print(f"  ⚠️ Classification call failed ({type(e).__name__}: {e}); "
              f"defaulting all {len(steps)} steps to Reasoning so the patient is not skipped.")
        parsed = None

    by_id = {}
    if parsed is not None:
        for a in (parsed.assessments if parsed else []) or []:
            if a.step_id and a.step_id not in by_id:
                by_id[a.step_id] = a

    assessments = []
    for i in range(1, len(steps) + 1):
        a = by_id.get(i)
        if a is None:
            a = StepClassification(step_id=i, classification=EfficiencyCategory.REASONING,
                                   rationale="missing from judge output; defaulted to Reasoning")
        assessments.append(a)

    for i, assessment in enumerate(assessments, 1):
        print(f"Assessing Step {i}: {steps[i - 1][:100]}...")
        print(f"  Classification: {assessment.classification.value}")
        print(f"  Rationale: {assessment.rationale}")
        print("-" * 50)

    return assessments


def calculate_res(assessments: List[StepClassification]) -> Tuple[float, int, int, int, Dict[str, Dict]]:
    """Compute the Reasoning Efficiency Score and category statistics."""
    print("\n=== starting RES calculation ===")
    reasoning_count = sum(1 for ass in assessments if ass.classification == EfficiencyCategory.REASONING)
    citation_count = sum(1 for ass in assessments if ass.classification == EfficiencyCategory.CITATION)
    total_steps = len(assessments)
    res_score = (reasoning_count / total_steps * 100) if total_steps else 0

    category_counts = Counter(ass.classification.value for ass in assessments)
    category_stats = {
        category: {"count": count, "percentage": (count / total_steps * 100) if total_steps else 0}
        for category, count in category_counts.items()
    }

    print(f"Reasoning Steps: {reasoning_count} | Citation: {citation_count} | "
          f"Total: {total_steps} | RES: {res_score:.1f}%")
    return res_score, reasoning_count, citation_count, total_steps, category_stats
