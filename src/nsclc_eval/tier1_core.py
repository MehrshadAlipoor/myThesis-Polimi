"""Tier 1 Core — Reasoning Efficiency Score (RES).

Decomposes the raw agent reasoning into atomic steps, classifies each step as
*Citation / Repetition / Reasoning / Redundancy* and computes the
Reasoning Efficiency Score:

    RES = (reasoning_steps / total_steps) x 100

Step classification is performed in parallel with a ``ThreadPoolExecutor``.
"""

from __future__ import annotations

import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple

from . import config
from .models import EfficiencyCategory, StepClassification
from .prompts import REASONING_PROMPT, REFORMAT_PROMPT


def decompose_reasoning(raw_reasoning: str) -> str:
    """Ask the judge LLM to reorganize raw reasoning into atomic steps."""
    response_reasoning = config.client.chat.completions.create(
        model=config.JUDGE_MODEL,
        messages=[
            {"role": "system", "content": REFORMAT_PROMPT},
            {"role": "user", "content": f"Final report: {raw_reasoning}"},
        ],
        temperature=config.TEMPERATURE,
        seed=config.SEED,
    )
    atomic_steps = response_reasoning.choices[0].message.content
    print("\n=== Atomic Steps from Rationale ===\n")
    print(atomic_steps)
    return atomic_steps


def _extract_step_texts(step_blob: str) -> List[str]:
    """Parse the LLM's reformatted output into a list of step texts."""
    if not step_blob:
        return []

    step_header_pattern = re.compile(
        r"^(?:[-*\s\u2022]*)?(?:<|\[)?Step\s*\d+\s*(?:>|\]|:|\)|\.)? *(.*)$",
        re.IGNORECASE,
    )
    numeric_pattern = re.compile(r"^(?:[-*\s\u2022]*)\d+[\s\)\.:\-]+\s*(.*)$")

    extracted_steps = []
    current_step_lines = []
    lines = [line.strip() for line in str(step_blob).replace(
        "\r", "\n").split("\n") if line.strip()]
    has_explicit_steps = any(step_header_pattern.match(
        l) or numeric_pattern.match(l) for l in lines)

    if has_explicit_steps:
        for line in lines:
            match = step_header_pattern.match(
                line) or numeric_pattern.match(line)
            if match:
                if current_step_lines:
                    extracted_steps.append(
                        " ".join(current_step_lines).strip())
                    current_step_lines = []
                remainder = match.group(1).strip()
                if remainder:
                    current_step_lines.append(remainder)
            else:
                if current_step_lines:
                    current_step_lines.append(line)
        if current_step_lines:
            extracted_steps.append(" ".join(current_step_lines).strip())
    else:
        for line in lines:
            clean_line = re.sub(r"^[-*\u2022\s>+]+", "", line).strip()
            if clean_line:
                extracted_steps.append(clean_line)

    return [step for step in extracted_steps if step]


def classify_reasoning_steps(atomic_steps: str) -> List[StepClassification]:
    """Classify each atomic step using the judge LLM (in parallel)."""
    steps = _extract_step_texts(atomic_steps)
    print(f"\nFound {len(steps)} steps to assess.\n")

    def process_step(i: int, step_text: str):
        # No printing here to avoid interleaved step logs.
        res = config.client.beta.chat.completions.parse(
            model=config.JUDGE_MODEL,
            messages=[
                {"role": "system", "content": REASONING_PROMPT},
                {"role": "user", "content": f"Step: {step_text}"},
            ],
            response_format=StepClassification,
            temperature=config.TEMPERATURE,
            seed=config.SEED,
        )
        assessment = res.choices[0].message.parsed
        return i, assessment

    assessments = [None] * len(steps)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(process_step, i, step_text)
                   for i, step_text in enumerate(steps, 1)]
        for future in futures:
            i, assessment = future.result()
            assessments[i - 1] = assessment

    for i, assessment in enumerate(assessments, 1):
        print(f"Assessing Step {i}: {steps[i - 1][:100]}...")
        print(f"  Classification: {assessment.classification.value}")
        print(f"  Rationale: {assessment.rationale}")
        print("-" * 50)

    return assessments


def calculate_res(
    assessments: List[StepClassification],
) -> Tuple[float, int, int, int, Dict[str, Dict]]:
    """Compute the Reasoning Efficiency Score and category statistics."""
    print("\n=== starting RES calculation ===")
    reasoning_count = sum(
        1 for ass in assessments if ass.classification == EfficiencyCategory.REASONING
    )
    citation_count = sum(
        1 for ass in assessments if ass.classification == EfficiencyCategory.CITATION
    )
    total_steps = len(assessments)
    res_score = (reasoning_count / total_steps * 100) if total_steps else 0

    category_counts = Counter(ass.classification.value for ass in assessments)
    category_stats = {
        category: {
            "count": count,
            "percentage": (count / total_steps * 100) if total_steps else 0,
        }
        for category, count in category_counts.items()
    }

    print(
        f"Reasoning Steps: {reasoning_count} | Citation: {citation_count} | "
        f"Total: {total_steps} | RES: {res_score:.1f}%"
    )
    return res_score, reasoning_count, citation_count, total_steps, category_stats
