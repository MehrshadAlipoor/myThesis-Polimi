"""Tier 2 — Clinical Effectiveness (binary IO vs IOCT accuracy).

Extracts the predicted treatment class (Immunotherapy alone vs Immunotherapy
+ Chemotherapy) from the agent's final decision and compares it against the
ground-truth label from the clinical cohort.
"""

from __future__ import annotations

from typing import Any, Tuple

from . import config
from .models import AccuracyExtraction
from .prompts import TREATMENT_EXTRACTION_PROMPT


def _label_text(binary_label: Any) -> str:
    if binary_label == 1:
        return "Immunotherapy + Chemotherapy (IOCT)"
    if binary_label == 0:
        return "Immunotherapy alone (IO)"
    return "Unknown"


def _extract_binary_prediction(model_decision: str) -> AccuracyExtraction:
    """Ask the judge LLM to classify the predicted treatment as IO (0) or IOCT (1)."""
    response = config.client.beta.chat.completions.parse(
        model=config.JUDGE_MODEL,
        messages=[
            {"role": "system", "content": TREATMENT_EXTRACTION_PROMPT},
            {"role": "user", "content": f"Predicted treatment: {model_decision}"},
        ],
        response_format=AccuracyExtraction,
        temperature=config.TEMPERATURE,
        seed=config.SEED,
    )
    return response.choices[0].message.parsed


def evaluate_clinical_effectiveness(
    model_decision: str, ground_truth_value: Any
) -> Tuple[Any, str, Any, str]:
    """Evaluate the binary clinical accuracy of the agent's treatment decision."""
    print("\n=== Evaluating Clinical Effectiveness ===")
    extracted_therapy = _extract_binary_prediction(model_decision)
    predicted_label = extracted_therapy.predicted_label
    ground_truth_label = _label_text(ground_truth_value)
    binary_accuracy = (
        1
        if predicted_label is not None and ground_truth_value is not None
        and predicted_label == ground_truth_value
        else 0
    )
    accuracy_result = "Correct" if binary_accuracy == 1 else "Wrong"
    print(f"  Predicted: {predicted_label} => {extracted_therapy.category.value}")
    print(f"  Ground Truth: {ground_truth_value} => {ground_truth_label}")
    print(f"  Binary Accuracy: {binary_accuracy} ({accuracy_result})")
    return binary_accuracy, accuracy_result, predicted_label, ground_truth_label
