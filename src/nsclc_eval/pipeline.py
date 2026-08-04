"""Pipeline orchestration.

Coordinates the full evaluation flow:

1. ``_build_evaluation_payload`` — runs Tier 1 Core (RES) sequentially, then
   Tier 1 Extended + Tier 2 metrics in parallel via a ``ThreadPoolExecutor``.
2. ``_evaluate_file_by_mode`` — per-patient wrapper.
3. ``run_evaluation`` — single (judge, orchestrator) batch with aggregation.
4. ``run_all_patient_evaluations`` — full all-to-all judge x orchestrator matrix,
   restarting the Docker ``llama-cpp-server`` whenever the judge model changes
   (VRAM management).
"""

from __future__ import annotations

import json
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional

from . import config
from .data_loading import load_patient_data
from .tier1_core import (
    _extract_step_texts,
    calculate_res,
    classify_reasoning_steps,
    decompose_reasoning,
)
from .tier1_extended import (
    _extract_retrieved_guidelines,
    evaluate_factuality,
    evaluate_faithfulness,
    evaluate_guideline_adherence,
    evaluate_reasoning_completeness_llm,
    evaluate_treatment_plan_completeness,
)
from .tier2_clinical import _extract_binary_prediction, evaluate_clinical_effectiveness
from .utils import (
    _get_run_output_dir,
    _sanitize_filename,
    _write_error_log,
    build_file_index,
    save_evaluation_results,
)

DEFAULT_EVAL_FLAGS = {
    "tier1_core": True,
    "tier1_reasoning_completeness": True,
    "tier1_guideline_adherence": True,
    "tier1_factuality": True,
    "tier1_faithfulness": True,
    "tier2_binary_accuracy": True,
    "tier2_treatment_completeness": True,
}


def _build_evaluation_payload(
    patient_id: str,
    i3lung_id: str,
    orchestrator_model_name: str,
    raw_reasoning: str,
    raw_decision: str,
    ground_truth_value: Any,
    mode: str,
    run_started_at: Optional[datetime],
    source_name: str,
    source_path: str,
    pipeline_data: Optional[Dict] = None,
    step5_output: Optional[str] = None,
    eval_flags: Optional[Dict] = None,
):
    if eval_flags is None:
        eval_flags = dict(DEFAULT_EVAL_FLAGS)

    assessments = []
    steps = []
    res_score = None
    reasoning_count = citation_count = total_steps = 0
    category_stats = {}
    binary_accuracy = None
    accuracy_result = ground_truth_text = None
    extracted_therapy = None
    predicted_label = None
    gar_result = completeness_result = factuality_result = faithfulness_result = treatment_completeness = None

    t0 = time.time()

    if eval_flags.get("tier1_core"):
        atomic_steps = decompose_reasoning(raw_reasoning)
        assessments = classify_reasoning_steps(atomic_steps)
        steps = _extract_step_texts(atomic_steps)
        res_score, reasoning_count, citation_count, total_steps, category_stats = calculate_res(assessments)
    else:
        if any(
            eval_flags.get(f)
            for f in [
                "tier1_reasoning_completeness",
                "tier1_guideline_adherence",
                "tier1_factuality",
                "tier1_faithfulness",
            ]
        ):
            atomic_steps = decompose_reasoning(raw_reasoning)
            steps = _extract_step_texts(atomic_steps)

    with ThreadPoolExecutor(max_workers=1) as executor:
        f_comp = f_gar = f_fact = f_faith = f_tier2 = f_treat_comp = None
        step3_output = _extract_retrieved_guidelines(pipeline_data)

        if pipeline_data and steps:
            print("\n--- Tier 1 Extended Metrics ---")
            if eval_flags.get("tier1_guideline_adherence") and step3_output:
                f_gar = executor.submit(evaluate_guideline_adherence, steps, step3_output)
            if eval_flags.get("tier1_reasoning_completeness"):
                f_comp = executor.submit(evaluate_reasoning_completeness_llm, steps, pipeline_data)
            if eval_flags.get("tier1_factuality"):
                f_fact = executor.submit(evaluate_factuality, steps, pipeline_data)
            if eval_flags.get("tier1_faithfulness"):
                f_faith = executor.submit(evaluate_faithfulness, steps, raw_decision)

        if eval_flags.get("tier2_treatment_completeness") and step5_output:
            f_treat_comp = executor.submit(evaluate_treatment_plan_completeness, step5_output, pipeline_data)

        if eval_flags.get("tier2_binary_accuracy"):
            def do_tier2():
                bin_acc, acc_res, p_label, gt_text = evaluate_clinical_effectiveness(
                    raw_decision, ground_truth_value
                )
                ext_therapy = _extract_binary_prediction(raw_decision)
                return bin_acc, acc_res, p_label, gt_text, ext_therapy

            f_tier2 = executor.submit(do_tier2)

        if f_comp:
            completeness_result = f_comp.result()
        if f_gar:
            gar_result = f_gar.result()
        if f_fact:
            factuality_result = f_fact.result()
        if f_faith:
            faithfulness_result = f_faith.result()
        if f_treat_comp:
            treatment_completeness = f_treat_comp.result()

        if f_tier2:
            binary_accuracy, accuracy_result, predicted_label, ground_truth_text, extracted_therapy = f_tier2.result()

    duration_sec = time.time() - t0

    evaluation_data = {
        "evaluation_settings": {
            "evaluation_date": datetime.now().isoformat(),
            "evaluation_duration_min": (
                (datetime.now() - run_started_at).total_seconds() / 60 if run_started_at else None
            ),
            "patient_processing_time_sec": duration_sec,
            "mode": mode,
            "eval_flags": eval_flags,
        },
        "source_file": {
            "name": source_name,
            "path": source_path,
            "patient_id": {"cartella": patient_id, "i3lung": i3lung_id},
        },
        "models": {"orchestrator": orchestrator_model_name, "judge": config.JUDGE_MODEL},
        "metrics_tier1": {"res": res_score, "tier1_category_stats": category_stats},
        "assessments": [
            {
                "step": i + 1,
                "claim": steps[i] if i < len(steps) else "",
                "classification": ass.classification.value,
                "rationale": ass.rationale,
            }
            for i, ass in enumerate(assessments)
        ]
        if assessments
        else [],
        "metrics_tier1_guideline_adherence": gar_result,
        "guideline_ground_truth_text": step3_output if step3_output else None,
        "metrics_tier1_completeness": completeness_result,
        "metrics_tier1_factuality": factuality_result,
        "metrics_tier1_faithfulness": faithfulness_result,
        "metrics_tier2_treatment_completeness": treatment_completeness,
        "metrics_tier2": (
            {
                "predicted_label": predicted_label,
                "ground_truth": ground_truth_text,
                "binary_accuracy": binary_accuracy,
                "accuracy_result": accuracy_result,
                "rationale": extracted_therapy.rationale if extracted_therapy else "",
            }
            if eval_flags.get("tier2_binary_accuracy")
            else {}
        ),
        "original_step5_log": {"raw_reasoning": raw_reasoning, "raw_decision": raw_decision},
    }

    return (
        evaluation_data,
        gar_result,
        res_score,
        reasoning_count,
        citation_count,
        total_steps,
        predicted_label,
        binary_accuracy,
        accuracy_result,
        completeness_result,
        factuality_result,
        faithfulness_result,
        treatment_completeness,
    )


def _evaluate_file_by_mode(
    file_path: str,
    gt_mapping: Dict,
    mode: str,
    save_results: bool = True,
    verbose: bool = True,
    run_date: Optional[str] = None,
    run_timestamp: Optional[str] = None,
    run_started_at: Optional[datetime] = None,
    judge_model: Optional[str] = None,
    eval_flags: Optional[Dict] = None,
) -> Optional[Dict]:
    patient_record, model_decision, pipeline_data = load_patient_data(file_path)
    ground_truth_value = gt_mapping.get(str(patient_record.i3lung_id).strip())
    if ground_truth_value is None:
        return None

    step5_logs = pipeline_data.get("step5_logs", [])
    step5_output = (
        step5_logs[0].get("output")
        if (step5_logs and isinstance(step5_logs[0], dict))
        else model_decision
    )

    source_name = os.path.basename(file_path)
    active_judge_model = judge_model or config.JUDGE_MODEL

    (
        evaluation_data,
        gar_result,
        res_score,
        reasoning_count,
        citation_count,
        total_steps,
        predicted_label,
        binary_accuracy,
        accuracy_result,
        completeness_result,
        factuality_result,
        faithfulness_result,
        treatment_completeness,
    ) = _build_evaluation_payload(
        patient_id=patient_record.patient_id,
        i3lung_id=patient_record.i3lung_id,
        orchestrator_model_name=patient_record.orchestration_model_name,
        raw_reasoning=patient_record.raw_reasoning,
        raw_decision=model_decision,
        ground_truth_value=ground_truth_value,
        mode=mode,
        run_started_at=run_started_at,
        source_name=source_name,
        source_path=file_path,
        pipeline_data=pipeline_data,
        step5_output=step5_output,
        eval_flags=eval_flags,
    )

    saved_path = None
    if save_results:
        saved_path = save_evaluation_results(
            evaluation_data,
            patient_record.patient_id,
            run_date,
            run_timestamp,
            orchestrator_model=patient_record.orchestration_model_name,
            judge_model=active_judge_model,
        )
        if verbose:
            print(f"✅ Saved to {saved_path}")

    return {
        "patient_id": patient_record.patient_id,
        "i3lung_id": patient_record.i3lung_id,
        "orchestrator_model": patient_record.orchestration_model_name,
        "judge_model": active_judge_model,
        "res_score": res_score,
        "total_steps": total_steps,
        "gar_score": gar_result.get("gar_score", 0) if gar_result else None,
        "reasoning_steps": reasoning_count,
        "citation_steps": citation_count,
        "predicted_label": predicted_label,
        "ground_truth_label": ground_truth_value,
        "binary_accuracy": binary_accuracy,
        "accuracy_result": accuracy_result,
        "reasoning_completeness_score": (
            completeness_result.get("completeness_score", 0) if completeness_result else None
        ),
        "treatment_completeness_score": (
            treatment_completeness.get("completeness_score", 0)
            if treatment_completeness
            else None
        ),
        "factuality_score": factuality_result.get("factuality_score", 0) if factuality_result else None,
        "faithfulness_score": (
            faithfulness_result.get("faithfulness_score", 0) if faithfulness_result else None
        ),
        "source_file": file_path,
        "saved_path": saved_path,
    }


def run_evaluation(
    file_paths: List[str],
    gt_mapping: Dict,
    mode: str = "both",
    judge_model: Optional[str] = None,
    orchestrator_model: Optional[str] = None,
    error_log_path: Optional[str] = None,
    summary_log_path: Optional[str] = None,
    save_results: bool = True,
    verbose: bool = True,
    run_date: Optional[str] = None,
    run_timestamp: Optional[str] = None,
    eval_flags: Optional[Dict] = None,
) -> List[Dict]:
    """Run a single (judge, orchestrator) batch and aggregate results."""
    if run_date is None:
        run_date = datetime.now().strftime("%Y-%m-%d")
    if run_timestamp is None:
        run_timestamp = datetime.now().strftime("%H-%M-%S")
    run_started_at = datetime.now()
    active_judge_model = judge_model or config.JUDGE_MODEL
    active_orch_model = orchestrator_model or "unknown_orchestrator"
    results_summary = []
    error_records = []

    def process_file(idx: int, file_path: str):
        try:
            if verbose:
                print(f"\nEvaluating file #{idx}/{len(file_paths)} from {file_path}")
                print(f"  Judge: {active_judge_model} | Orchestrator: {active_orch_model}\n")
            result = _evaluate_file_by_mode(
                file_path,
                gt_mapping,
                mode,
                save_results=save_results,
                verbose=verbose,
                run_date=run_date,
                run_timestamp=run_timestamp,
                run_started_at=run_started_at,
                judge_model=active_judge_model,
                eval_flags=eval_flags,
            )
            if result:
                result["orchestrator_model"] = active_orch_model
                result["judge_model"] = active_judge_model
                return result
        except Exception as exc:
            err = {
                "file_path": file_path,
                "orchestrator_model": active_orch_model,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
                "failed_at": datetime.now().isoformat(),
            }
            if verbose:
                print(f"⚠️ Skipping {file_path}: {type(exc).__name__}: {exc}")
            return err
        return None

    # max_workers=1 prevents patient logs from interleaving
    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = {executor.submit(process_file, idx, fp): fp for idx, fp in enumerate(file_paths, 1)}
        for future in as_completed(futures):
            res = future.result()
            if res is None:
                continue
            if "error_type" in res:
                error_records.append(res)
            else:
                results_summary.append(res)

    def _safe_avg(key: str):
        vals = [r[key] for r in results_summary if r.get(key) is not None]
        return sum(vals) / max(1, len(vals)) if vals else None

    avg_res = _safe_avg("res_score")
    avg_acc = _safe_avg("binary_accuracy")
    avg_gar = _safe_avg("gar_score")
    avg_reasoning_comp = _safe_avg("reasoning_completeness_score")
    avg_treatment_comp = _safe_avg("treatment_completeness_score")
    avg_fact = _safe_avg("factuality_score")
    avg_faith = _safe_avg("faithfulness_score")

    if verbose:
        print(f"\n{'='*60}")
        print(f"Batch Summary: Judge={active_judge_model} | Orch={active_orch_model}")
        print(f"{'='*60}")
        for res in results_summary:
            print(
                f"  Patient: {res['patient_id']} | "
                f"RES: {res.get('res_score', 'N/A')} | Acc: {res.get('binary_accuracy', 'N/A')} | "
                f"GAR: {res.get('gar_score', 'N/A')}% | Reasoning Compl: {res.get('reasoning_completeness_score', 'N/A')} | Treatment Compl: {res.get('treatment_completeness_score', 'N/A')} | "
                f"Fact: {res.get('factuality_score', 'N/A')} | Faith: {res.get('faithfulness_score', 'N/A')}"
            )
        if results_summary:
            _f = lambda v: f"{v:.1f}" if v is not None else "N/A"
            print(
                f"\nAverages: RES={_f(avg_res)}% | Acc={_f(avg_acc)} | "
                f"GAR={_f(avg_gar)}% | Reasoning Compl={_f(avg_reasoning_comp)}% | Treatment Compl={_f(avg_treatment_comp)}% | Fact={_f(avg_fact)}% | Faith={_f(avg_faith)}"
            )
        print(f"Evaluated: {len(results_summary)} | Failed: {len(error_records)}")
        print(f"Duration: {(datetime.now() - run_started_at).total_seconds() / 60:.2f} min")

    summary_data = {
        "evaluation_date": datetime.now().isoformat(),
        "mode": mode,
        "judge_model": active_judge_model,
        "orchestrator_model": active_orch_model,
        "total_files_considered": len(file_paths),
        "successful_files": len(results_summary),
        "failed_files": len(error_records),
        "average_res": avg_res,
        "average_binary_accuracy": avg_acc,
        "avg_gar": avg_gar,
        "avg_reasoning_completeness": avg_reasoning_comp,
        "avg_treatment_completeness": avg_treatment_comp,
        "avg_factuality_score": avg_fact,
        "avg_faithfulness_score": avg_faith,
        "results": results_summary,
        "duration_min": (datetime.now() - run_started_at).total_seconds() / 60,
    }

    if save_results:
        output_dir = _get_run_output_dir(run_date, run_timestamp, judge_model=active_judge_model)
        sanitized_orch = _sanitize_filename(active_orch_model)
        sanitized_judge = _sanitize_filename(active_judge_model)
        if summary_log_path is None:
            summary_log_path = f"{output_dir}/{sanitized_orch}_{sanitized_judge}_overall_summary.json"
        os.makedirs(os.path.dirname(summary_log_path), exist_ok=True)
        with open(summary_log_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=4, ensure_ascii=False)
        if verbose:
            print(f"✅ Summary saved to {summary_log_path}")
        if error_records:
            if error_log_path is None:
                error_log_path = f"{output_dir}/{sanitized_orch}_{sanitized_judge}_errors.json"
            _write_error_log(error_log_path, error_records)

    return results_summary


def run_all_patient_evaluations(
    data_dir: str,
    gt_mapping: Dict,
    judge_models: List[str],
    mode: str = "both",
    orchestrators_to_evaluate: Optional[List[str]] = None,
    max_files_per_batch: Optional[int] = None,
    save_results: bool = True,
    verbose: bool = True,
    eval_flags: Optional[Dict] = None,
    restart_server: bool = True,
) -> List[Dict]:
    """Run the full all-to-all judge x orchestrator evaluation matrix.

    The Docker ``llama-cpp-server`` is restarted only when the judge model
    changes (VRAM management). Set ``restart_server=False`` when running
    against an external/always-up endpoint or a mock client.
    """
    orch_to_files, discovered_orchestrators = build_file_index(data_dir)
    active_orchestrators = orchestrators_to_evaluate or discovered_orchestrators

    if verbose:
        print(f"\nOrchestrators: {active_orchestrators}")
        print(f"Judges: {judge_models}")
        total_batches = len(judge_models) * len(active_orchestrators)
        print(f"Total batches (judge x orch): {total_batches}\n")

    run_date = datetime.now().strftime("%Y-%m-%d")
    run_timestamp = datetime.now().strftime("%H-%M-%S")
    all_results = []
    batch_num = 0

    t_all_start = time.time()

    current_judge = None  # VRAM flushing

    for judge_model in judge_models:
        # ── VRAM FLUSH: restart container only when judge changes ──
        if restart_server and judge_model != current_judge:
            print(f"\n🔄 Swapping judge model → {judge_model}")
            config.restart_llama_server(judge_model)
            current_judge = judge_model
        # ────────────────────────────────────────────────────────────

        for orch_model in active_orchestrators:
            batch_files = orch_to_files.get(orch_model, [])
            if not batch_files:
                continue
            if max_files_per_batch is not None:
                batch_files = batch_files[:max_files_per_batch]
            batch_num += 1
            if verbose:
                print(f"\n{'='*70}")
                print(f"BATCH {batch_num}/{total_batches}: Judge={judge_model} | Orch={orch_model}")
                print(f"Patients in batch: {len(batch_files)}")
                print(f"{'='*70}")

            config.JUDGE_MODEL = judge_model

            results = run_evaluation(
                file_paths=batch_files,
                gt_mapping=gt_mapping,
                mode=mode,
                judge_model=judge_model,
                orchestrator_model=orch_model,
                save_results=save_results,
                verbose=verbose,
                run_date=run_date,
                run_timestamp=run_timestamp,
                eval_flags=eval_flags,
            )
            all_results.extend(results)

    if verbose:
        print(f"\n{'='*70}")
        print(f"ALL DONE. Total evaluations: {len(all_results)}")
        print(f"Total time elapsed: {(time.time() - t_all_start)/60:.2f} minutes")
        print(f"{'='*70}")
    return all_results
