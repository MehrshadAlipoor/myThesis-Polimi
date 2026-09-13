"""Prompt templates for the NSCLC therapy-agent evaluation framework.

Each prompt is used with a different LLM-as-a-Judge metric:

* ``REFORMAT_PROMPT`` — decomposes raw agent reasoning into atomic steps.
* ``BATCHED_CLASSIFICATION_PROMPT`` — classifies **all** steps at once (RES).
* ``TREATMENT_EXTRACTION_PROMPT`` — extracts the binary IO vs IOCT class (Tier 2).
* ``FACTUALITY_PROMPT`` — verifies factual claims against the patient record.
* ``OUTPUT_FACTUALITY_PROMPT`` — same audit over the final treatment output.
* ``FAITHFULNESS_PROMPT`` — checks internal consistency of the reasoning chain.
* ``REASONING_COMPLETENESS_PROMPT`` — 8-item clinical checklist over reasoning steps.
* ``TREATMENT_PLAN_COMPLETENESS_PROMPT`` — 8-item clinical checklist over the treatment plan.
* ``GUIDELINE_ADHERENCE_PROMPT`` — checks each step against retrieved ESMO/ASCO guidelines (GAR).

The ``ACTIVE_PROMPTS`` / ``P()`` hook lets the prompt-sensitivity runner inject
paraphrased variants (v1/v2) without touching the metric code.
"""

REFORMAT_PROMPT = """
# Task Overview
Given a the final report of an nsclc agentic ai with multiple reasoning steps [Text to be Organized], reorganize it into clearly structured steps, separated by newline characters.

# Organization Requirements
Convert the original raw reasoning of ai agent system into a clear, structured reasoning process, while ensuring:
- All original key information is preserved, but if multiple sentences discuss the same topic or serve the same logical reasoning purpose, they can be combined into one reasoning step.
- No new explanations or reasoning are added.
- No steps are omitted.

# Requirements
- Each step must be atomic (one conclusion per step).
- There should be no content repetition between steps.
- The final answer determination is also considered a step in the logical reasoning.

# Output Requirements
1.There should be no newline characters within each step, and each step should be separated by a single newline character.
2.For highly repetitive reasoning patterns, output them as a single step.
3.Output a maximum of 10 steps.

# Output Format
<Step 1> Content of this reasoning step...
<Step 2> Content of this reasoning step...
...
<Step n> Content of this reasoning step...

Below is the text that needs to be reorganized into reasoning steps:
[Text to be Organized]
"""

BATCHED_CLASSIFICATION_PROMPT = """
# Task Description
You will receive a numbered list of reasoning steps produced by a clinical AI agent for NSCLC therapy selection.
Classify EVERY step into exactly one of:
1. Citation: Restatement of record info without new reasoning.
2. Repetition: Repetition of previous steps without advancing process.
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

TREATMENT_EXTRACTION_PROMPT = '''
# Task Description
As a clinical data extraction system, analyze the provided "Proposed Treatment Plan" and classify it into one of two categories:
- **Class 0 (IO - Immunotherapy Alone):** ONLY monoclonal antibodies (e.g., Pembrolizumab, Cemiplimab, Atezolizumab) without cytotoxic chemotherapy.
- **Class 1 (IOCT - Immunotherapy + Chemotherapy):** Immunotherapy combined with cytotoxic agents (Carboplatin, Pemetrexed, Paclitaxel).

# Reference Knowledge
1. IOCT (Output 1): Carboplatin + Pemetrexed + Immunotherapy; Carboplatin + Paclitaxel + Immunotherapy
2. IO (Output 0): Pembrolizumab monotherapy; Cemiplimab monotherapy; Atezolizumab monotherapy

# Constraints
- If ambiguous but mentions "chemotherapy" or cytotoxic drugs alongside immunotherapy, output 1.
- If ONLY immunotherapy agents are listed, output 0.

# Output Requirement
Output ONLY the digit "0" or "1".
'''

# === TIER 1 EXTENDED PROMPTS ===

FACTUALITY_PROMPT = """
# Task Description
You are a clinical factuality auditor for NSCLC therapy AI agents.

You will receive:
1. **PATIENT_DATA**: The actual clinical record fields of a patient (structured key-value pairs).
2. **REASONING_STEPS**: Numbered reasoning steps produced by an AI agent.

# Your Task
- Scan the reasoning steps and identify every **factual claim** the agent makes about the patient's clinical data.
- For each claim, check whether it matches PATIENT_DATA.
- Classify each claim's verdict as: "correct", "incorrect", or "unsupported".
- Calculate `factuality_score` as `(correct_claims / total_claims) * 100`. If no claims found, set to 100.0.

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

OUTPUT_FACTUALITY_PROMPT = """
# Task Description
You are a clinical factuality auditor for NSCLC therapy AI agents.

You will receive:
1. **PATIENT_DATA**: The actual clinical record fields of a patient (structured key-value pairs).
2. **FINAL_OUTPUT** / **TREATMENT_PLAN**: The agent's final output / treatment plan text.

# Your Task
- Scan the final output and identify every **factual claim** the agent makes about the patient's clinical data.
- For each claim, check whether it matches PATIENT_DATA.
- Classify each claim's verdict as: "correct", "incorrect", or "unsupported".
- Calculate `factuality_score` as `(correct_claims / total_claims) * 100`. If no claims found, set to 100.0.

# Important Rules
- Only extract claims about patient-specific clinical facts, NOT general medical knowledge.
- If a field has multiple values or aliases (e.g., "WT" and "Wild-type"), treat them as equivalent.
- The final output is not step-decomposed, so use step_number = 0 for every claim.

# Output Format (JSON)
Each claim MUST have exactly these 5 fields:
- "step_number": integer (0 for the final output)
- "claim": string (the factual claim text)
- "referenced_field": string (which clinical field, e.g. "histology", "pdl1_tps")
- "actual_value": string (the actual value from PATIENT_DATA)
- "verdict": string (one of: "correct", "incorrect", "unsupported")

Example output:
{"claims": [{"step_number": 0, "claim": "Patient has adenocarcinoma", "referenced_field": "histology", "actual_value": "Adenocarcinoma", "verdict": "correct"}], "total_claims": 1, "correct_claims": 1, "incorrect_claims": 0, "unsupported_claims": 0, "factuality_score": 100.0}
"""

FAITHFULNESS_PROMPT = """
# Task Description
You are a reasoning consistency auditor for NSCLC therapy AI agents.

You will receive:
1. **REASONING_STEPS**: Numbered reasoning steps from an AI agent for NSCLC therapy selection.
2. **FINAL_RECOMMENDATION**: The agent's final therapy recommendation text.

# Your Task
1. **Self-contradictions**: Check if any step contradicts another step.
2. **Conclusion follows**: Check if the final recommendation logically follows from the reasoning steps.
3. **Logical gaps**: Identify steps that make claims without supporting reasoning or that skip important logical links.

# Rating Scale (faithfulness_score)
- 5 = Fully consistent, no contradictions, conclusion follows from premises
- 4 = Minor gaps but no contradictions, conclusion generally follows
- 3 = One contradiction or one significant logical gap
- 2 = Multiple contradictions or conclusion doesn't match reasoning
- 1 = Severely inconsistent reasoning, multiple contradictions

# Output Format (JSON)
- "contradictions": list of {"step_a": int, "step_b": int, "description": string}
- "conclusion_follows": boolean
- "logical_gaps": list of {"step_number": int, "description": string}
- "faithfulness_score": integer (1-5)
- "summary": string
"""

REASONING_COMPLETENESS_PROMPT = '''
# Task Description
You are a clinical reasoning completeness auditor for NSCLC therapy AI agents.

You will receive:
1. REASONING_STEPS: Numbered reasoning steps from an AI agent.
2. PATIENT_DATA: The actual patient clinical record (structured key-value pairs).

# Your Task
For each item in the checklist below, determine if the reasoning steps
MENTION or DISCUSS this clinical factor. Extract the value the agent used
(if any). Provide a brief rationale explaining your judgment.

# Checklist (8 items)
1. Histology — lung cancer histological subtype
2. PD-L1 — PD-L1 expression level (TPS or category)
3. ECOG PS — performance status score
4. Comorbidities — any comorbid conditions discussed
5. Disease burden — extent of metastatic spread, tumor measurements,
   specific sites (brain, liver, bone, etc.)
6. IO biomarkers — actionable mutations (EGFR, ALK, ROS1, BRAF, KRAS, etc.)
7. Stage — TNM staging or overall clinical stage (I–IV)
8. CT/IO drug contraindications — autoimmune disease, ILD, IBD,
   immunosuppressive meds, corticosteroids

# Output Format (JSON)
{
  "items": [
    {
      "checklist_item": "histology",
      "mentioned": true,
      "extracted_value": "adenocarcinoma",
      "rationale": "Step 1 explicitly states adenocarcinoma",
      "step_numbers": [1, 3]
    }
  ]
}
'''

TREATMENT_PLAN_COMPLETENESS_PROMPT = '''
# Task Description
You are a clinical treatment plan completeness auditor for NSCLC therapy AI agents.

You will receive:
1. TREATMENT_PLAN: The final treatment recommendation output from an AI agent.
2. PATIENT_DATA: The actual patient clinical record (structured key-value pairs).

# Your Task
For each item in the checklist below, determine if the treatment plan output
MENTIONS or DISCUSSES this clinical factor. Extract the value used (if any).
Provide a brief rationale explaining your judgment.

# Checklist (8 items)
1. Histology — lung cancer histological subtype
2. PD-L1 — PD-L1 expression level (TPS or category)
3. ECOG PS — performance status score
4. Comorbidities — any comorbid conditions discussed
5. Disease burden — extent of metastatic spread, tumor measurements,
   specific sites (brain, liver, bone, etc.)
6. IO biomarkers — actionable mutations (EGFR, ALK, ROS1, BRAF, KRAS, etc.)
7. Stage — TNM staging or overall clinical stage (I–IV)
8. CT/IO drug contraindications — autoimmune disease, ILD, IBD,
   immunosuppressive meds, corticosteroids

# Output Format (JSON)
{
  "items": [
    {
      "checklist_item": "histology",
      "mentioned": true,
      "extracted_value": "adenocarcinoma",
      "rationale": "Plan references adenocarcinoma when justifying choice"
    }
  ]
}
'''

GUIDELINE_ADHERENCE_PROMPT = '''
# Task Description
You are an expert clinical auditor evaluating an AI agent's reasoning against established clinical guidelines.

You will receive:
1. GUIDELINES_GROUND_TRUTH: The compiled guideline recommendations, exclusions, and analysis for this specific patient.
2. REASONING_STEPS: Numbered reasoning steps produced by the AI agent.

# Your Task
Evaluate each reasoning step against the GUIDELINES_GROUND_TRUTH.
For each step, determine if it adheres to the guidelines (True/False):
- Set to `True` if the step correctly applies the guidelines.
- Set to `True` if the step is just stating patient facts (Neutral steps do not violate guidelines).
- Set to `False` ONLY if the step actively contradicts the guidelines, recommends an excluded therapy, or ignores a documented contraindication.

Provide a brief rationale for your judgment.
'''

CONCISE_NOTE = ("\n\n# Brevity Requirement\n"
                "Keep every rationale/explanation under 25 words. Never repeat input text verbatim.\n")
for _p in ["FACTUALITY_PROMPT", "OUTPUT_FACTUALITY_PROMPT", "FAITHFULNESS_PROMPT",
           "REASONING_COMPLETENESS_PROMPT", "TREATMENT_PLAN_COMPLETENESS_PROMPT",
           "GUIDELINE_ADHERENCE_PROMPT"]:
    globals()[_p] = globals()[_p] + CONCISE_NOTE


# ── prompt-override hook (sensitivity testing) ────────────────────────────────
# Metric functions read prompts via P("<CONSTANT>") so a variant can be injected
# per-run without touching pipeline code. ACTIVE_PROMPTS is swapped inside the
# sensitivity runner's `with use_prompt_overrides(...)` block.
ACTIVE_PROMPTS = {}


def P(name):
    return ACTIVE_PROMPTS.get(name, globals().get(name, ""))
