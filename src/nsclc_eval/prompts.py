"""Prompt templates for the NSCLC therapy-agent evaluation framework.

Each prompt is used with a different LLM-as-a-Judge metric:

* ``REFORMAT_PROMPT`` — decomposes raw agent reasoning into atomic steps.
* ``REASONING_PROMPT`` — classifies a step as Citation/Repetition/Reasoning/Redundancy (RES).
* ``TREATMENT_EXTRACTION_PROMPT`` — extracts the binary IO vs IOCT treatment class (Tier 2).
* ``FACTUALITY_PROMPT`` — verifies factual claims against the patient record.
* ``FAITHFULNESS_PROMPT`` — checks internal consistency of the reasoning chain.
* ``REASONING_COMPLETENESS_PROMPT`` — 8-item clinical checklist over reasoning steps.
* ``TREATMENT_PLAN_COMPLETENESS_PROMPT`` — 8-item clinical checklist over the treatment plan.
* ``GUIDELINE_ADHERENCE_PROMPT`` — checks each step against retrieved ESMO/ASCO guidelines (GAR).
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

REASONING_PROMPT = """
# Task Description
Analyze the current thinking step and classify it into:
1. Citation: Restatement of record info without new reasoning.
2. Repetition: Repetition of previous steps without advancing process.
3. Reasoning: Deriving new conclusions that move toward the correct answer.
4. Redundancy: New info that does not help reach the final answer.

# Note
When determining the type, ensure to fully consider the logical relationship and reasoning process between the current thinking step, previous thinking steps, the patient's medical record, and the reasoning goal. If the current thinking step corresponds to multiple types, select the most appropriate one based on its contribution to the reasoning goal. Maintain objectivity and accuracy in judgment, avoiding subjective assumptions.

# Output Requirements
- Only output your classification of the current thinking step, with possible values being "Citation| Repetition|Reasoning|Redundancy".
- Do not output any other content.

# Output Format
Choose one from "Citation", "Repetition", "Reasoning", and "Redundancy".

Now, please classify the following input based on the instructions above:
[Current Thinking Step]
[All Previous Thinking Steps]
[Known Patient Medical Record]
[Final Reasoning Goal]

Here's an example of the reasoning step and its classification:
Imagine a patient case where a 25-year-old presents with a high fever, a severe headache, and a stiff neck.
- ==_Step 1:_== The patient presents with a high fever, severe headache, and stiff neck.
- ==_Step 2:_== These specific symptoms point toward meningitis.
- ==_Step 3:_== The patient has a headache.
- ==_Step 4:_== Meningitis is caused by the inflammation of the brain's blood vessels only.
- ==_Step 5:_== The patient's eye color is brown.

- ==Step 1== (restating facts): Citation.
- ==Step 2== (effective new insights): Reasoning.
- ==Step 3== (repeating past conclusions): Repetition.
- ==Step 4== (new, but we don't know if it's true yet): Reasoning.
- ==Step 5== (irrelevant): Redundancy.
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
      "rationale": "Step 1 explicitly states the patient has adenocarcinoma",
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
      "rationale": "The treatment plan references adenocarcinoma when justifying the choice"
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
