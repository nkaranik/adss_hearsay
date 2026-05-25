#!/usr/bin/env python3
"""Robust ADSS evaluation suite.

Fixes:
- BaselinePrediction is always constructed with keyword arguments.
- Full ADSS predictions are checkpointed after every case.
- Final reports are written only after all required predictions are available.
- Can resume from artifacts/<case_id>_prediction.json with --resume.

Suites:
- hearsay / legalbench: LegalBench hearsay primary benchmark.
- contract_nli: LegalBench contract_nli_confidentiality_of_agreement secondary benchmark.
- all_legalbench: runs hearsay then contract_nli separately, no aggregation.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.helpers import load_config, setup_logging, ensure_api_key
from src.data.loader import load_examples
from src.data.models import (
    ADSSPrediction,
    BaselinePrediction,
    ClaimNode,
    Decision,
    ExtractionResult,
    HearsayExample,
    SolverType,
)
from src.pipeline.orchestrator import ADSSPipeline
from src.qbaf.solver import get_solver, make_decision
from src.qbaf.graph import build_qbaf
import src.utils.llm_client as _llm

from src.evaluation.metrics import evaluate_system, mcnemar_test, write_metrics_report
from src.evaluation.simulations import run_contestability_simulation, write_contestability_report
from src.evaluation.robustness import run_robustness_suite, write_robustness_report
from src.evaluation.error_analysis import analyse_errors, write_error_report

HEARSAY_DECISION_PROBLEM = "Is the statement described in this narrative hearsay under FRE 801(c)?"
CONTRACT_NLI_DECISION_PROBLEM = "Does the contract context entail the target confidentiality statement?"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--suite", choices=["hearsay", "legalbench", "contract_nli", "all_legalbench"], default="hearsay")
    p.add_argument("--split", default="test")
    p.add_argument("--max", type=int, default=None)
    p.add_argument("--out", default="artifacts/evaluation_suite")
    p.add_argument("--delay", type=float, default=1.5)
    p.add_argument("--backend", choices=["gemini", "lmstudio"], default=None)
    p.add_argument("--solver", default="df_quad", choices=["df_quad", "qe_semantics"])
    p.add_argument("--bootstrap-samples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-baselines", action="store_true")
    p.add_argument("--skip-contestability", action="store_true")
    p.add_argument("--skip-robustness", action="store_true")
    p.add_argument("--save-predictions", action="store_true")
    p.add_argument("--resume", action="store_true", help="Reuse artifacts/<case_id>_prediction.json when present.")
    p.add_argument("--no-fallback", action="store_true")
    return p.parse_args()


def lmstudio_max_tokens(cfg: dict) -> int:
    return min(int(cfg.get("lmstudio", {}).get("max_tokens", 260000)), 260000)


def backend_max_tokens(cfg: dict, baseline: bool = False) -> int:
    if cfg.get("backend", "gemini").lower() == "lmstudio":
        return lmstudio_max_tokens(cfg)
    return min(int(cfg.get("gemini", {}).get("max_tokens", 4096)), 1024 if baseline else 4096)


def extract_yes_no(text: str) -> Decision:
    m = re.search(r"Answer:\s*(Yes|No)\b", text, re.IGNORECASE)
    if m:
        return Decision.YES if m.group(1).lower() == "yes" else Decision.NO
    low = text.lower()
    if "answer: yes" in low:
        return Decision.YES
    if "answer: no" in low:
        return Decision.NO
    if "not hearsay" in low or "non-hearsay" in low or "no hearsay" in low:
        return Decision.NO
    if "hearsay" in low:
        return Decision.YES
    return Decision.NO


def failed_baseline_prediction(item: Any, error: Exception, name: str) -> BaselinePrediction:
    ex = item[0] if isinstance(item, tuple) else item
    return BaselinePrediction(
        case_id=ex.case_id,
        baseline_name=name,
        decision=Decision.UNCERTAIN,
        raw_response=f"ERROR: {error}",
        gold_label=ex.label,
    )


def run_with_delay(fn: Callable[[Any], Any], items: list[Any], delay: float, fallback_factory=None) -> list[Any]:
    out: list[Any] = []
    for i, item in enumerate(items):
        try:
            out.append(fn(item))
        except Exception as exc:
            ex = item[0] if isinstance(item, tuple) else item
            print(f"[WARN] Evaluation call failed for {getattr(ex, 'case_id', i)}: {exc}")
            if fallback_factory is None:
                raise
            out.append(fallback_factory(item, exc))
        if delay and i < len(items) - 1:
            time.sleep(delay)
    return out


def baseline_zero_shot_cot(item: tuple[HearsayExample, str, str], cfg: dict) -> BaselinePrediction:
    ex, decision_problem, _domain = item
    prompt = (
        "You are an expert legal reasoner. Solve the decision problem for the case.\n"
        "Think step by step briefly. At the end output exactly one final line: Answer: Yes or Answer: No.\n\n"
        f"Decision problem: {decision_problem}\n\nCase:\n{ex.text}\n"
    )
    raw = _llm.call_llm(prompt, cfg, max_tokens=backend_max_tokens(cfg, baseline=True))
    return BaselinePrediction(
        case_id=ex.case_id,
        baseline_name="zero_shot_cot",
        decision=extract_yes_no(raw),
        raw_response=raw,
        gold_label=ex.label,
    )


def baseline_few_shot(item: tuple[HearsayExample, str, str], cfg: dict) -> BaselinePrediction:
    ex, decision_problem, domain = item
    if domain == "contract_nli":
        shots = (
            "Example 1\nDecision problem: Does the contract context entail the target confidentiality statement?\n"
            "Case:\nCONTRACT CONTEXT:\nThe Recipient shall keep all Confidential Information secret and shall not disclose it to third parties.\n\nTARGET STATEMENT:\nThe agreement requires the recipient to keep confidential information confidential.\nReasoning: The confidentiality clause directly entails the target statement.\nAnswer: Yes\n\n"
            "Example 2\nDecision problem: Does the contract context entail the target confidentiality statement?\n"
            "Case:\nCONTRACT CONTEXT:\nThe agreement describes payment terms and delivery dates but does not mention confidentiality.\n\nTARGET STATEMENT:\nThe agreement imposes a confidentiality obligation.\nReasoning: The target statement is not supported by the contract text.\nAnswer: No\n\n"
        )
    else:
        shots = (
            "Example 1\nDecision problem: Is the statement described in this narrative hearsay under FRE 801(c)?\n"
            "Case: A witness testifies that a friend said, 'The driver admitted he ran the red light.' The statement is offered to prove the driver ran the red light.\nReasoning: Out of court and offered for truth.\nAnswer: Yes\n\n"
            "Example 2\nDecision problem: Is the statement described in this narrative hearsay under FRE 801(c)?\n"
            "Case: A witness testifies that a warning sign said 'Wet floor' to show the defendant had notice, not to prove the floor was wet.\nReasoning: Offered for notice, not truth.\nAnswer: No\n\n"
        )
    prompt = (
        "You are an expert decision analyst. Follow the examples. End with Answer: Yes or Answer: No.\n\n"
        f"{shots}Target case\nDecision problem: {decision_problem}\nCase:\n{ex.text}\n"
    )
    raw = _llm.call_llm(prompt, cfg, max_tokens=backend_max_tokens(cfg, baseline=True))
    return BaselinePrediction(
        case_id=ex.case_id,
        baseline_name="few_shot_prompting",
        decision=extract_yes_no(raw),
        raw_response=raw,
        gold_label=ex.label,
    )


def non_symbolic_predictions(full_preds: list[ADSSPrediction], cfg: dict) -> list[BaselinePrediction]:
    from src.data.models import Stance
    preds: list[BaselinePrediction] = []
    for pred in full_preds:
        arg_map = {a.id: a for a in pred.extraction.arguments}
        sup = [s.tau for s in pred.strengths if arg_map.get(s.argument_id) and arg_map[s.argument_id].stance_to_claim == Stance.SUPPORT]
        att = [s.tau for s in pred.strengths if arg_map.get(s.argument_id) and arg_map[s.argument_id].stance_to_claim == Stance.ATTACK]
        if not sup and not att:
            sigma = 0.5
        else:
            agg_s = sum(sup) / len(sup) if sup else 0.0
            agg_a = sum(att) / len(att) if att else 0.0
            sigma = agg_s / (agg_s + agg_a + 1e-8)
        dec = Decision(make_decision(sigma, cfg)[0])
        preds.append(BaselinePrediction(
            case_id=pred.case_id,
            baseline_name="adss_without_symbolic_solver",
            decision=dec,
            raw_response=json.dumps({"sigma": sigma}),
            gold_label=pred.gold_label,
        ))
    return preds


def suite_items(suite: str, cfg: dict, split: str, max_n: int | None) -> list[tuple[HearsayExample, str, str]]:
    if suite in ("hearsay", "legalbench"):
        ds_cfg = cfg.get("datasets", {}).get("hearsay", cfg.get("dataset", {})).copy()
        if max_n:
            ds_cfg["max_samples"] = max_n
        examples = load_examples(ds_cfg, split)
        return [(ex, ds_cfg.get("decision_problem", HEARSAY_DECISION_PROBLEM), "hearsay") for ex in examples]
    if suite == "contract_nli":
        ds_cfg = cfg.get("datasets", {}).get("contract_nli", {}).copy()
        if max_n:
            ds_cfg["max_samples"] = max_n
        examples = load_examples(ds_cfg, split)
        return [(ex, ds_cfg.get("decision_problem", CONTRACT_NLI_DECISION_PROBLEM), "contract_nli") for ex in examples]
    raise ValueError(f"Unknown suite: {suite}")


def fallback_adss_prediction(ex: HearsayExample, error: Exception) -> ADSSPrediction:
    return ADSSPrediction(
        case_id=ex.case_id,
        input_text=ex.text,
        gold_label=ex.label,
        extraction=ExtractionResult(
            case_id=ex.case_id,
            input_text=ex.text,
            claim=ClaimNode(text="Fallback after pipeline error"),
            parse_error=str(error),
        ),
        sigma_phi=0.5,
        decision=Decision.UNCERTAIN,
    )


def load_prediction_artifact(path: Path) -> ADSSPrediction | None:
    try:
        txt = path.read_text(encoding="utf-8")
        if hasattr(ADSSPrediction, "model_validate_json"):
            return ADSSPrediction.model_validate_json(txt)
        return ADSSPrediction.parse_raw(txt)
    except Exception as exc:
        print(f"[WARN] Could not load existing artifact {path}: {exc}")
        return None


def run_full_adss_with_checkpoints(
    pipeline: ADSSPipeline,
    items: list[tuple[HearsayExample, str, str]],
    args: argparse.Namespace,
    out_dir: Path,
) -> list[ADSSPrediction]:
    preds: list[ADSSPrediction] = []
    artifact_dir = Path(pipeline.artifact_dir)
    partial_jsonl = out_dir / "full_adss_predictions_partial.jsonl"
    completed_path = out_dir / "full_adss_predictions.json"

    for i, item in enumerate(items):
        ex, decision_problem, _domain = item
        artifact_path = artifact_dir / f"{ex.case_id}_prediction.json"

        pred = None
        if args.resume and artifact_path.exists():
            pred = load_prediction_artifact(artifact_path)
            if pred is not None:
                print(f"[RESUME] Loaded {ex.case_id} from {artifact_path}")

        if pred is None:
            try:
                pred = pipeline.run_case(ex, decision_problem=decision_problem, save_artifacts=args.save_predictions)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                print(f"[WARN] Full ADSS failed for {ex.case_id}: {exc}")
                pred = fallback_adss_prediction(ex, exc)
                if args.save_predictions:
                    artifact_path.write_text(pred.model_dump_json(indent=2), encoding="utf-8")

        preds.append(pred)
        with partial_jsonl.open("a", encoding="utf-8") as f:
            f.write(pred.model_dump_json() + "\n")
        if args.delay and i < len(items) - 1:
            time.sleep(args.delay)

    completed_path.write_text(json.dumps([json.loads(p.model_dump_json()) for p in preds], indent=2), encoding="utf-8")
    return preds


def write_argument_diagnostics(predictions: list[ADSSPrediction], domains: dict[str, str], out_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    for pred in predictions:
        extraction = pred.extraction
        strengths = getattr(pred, "strengths", []) or []
        rels = getattr(extraction, "relations", []) or []
        n_raw = int(getattr(extraction, "n_raw_arguments", 0) or 0)
        n_kept = len(getattr(extraction, "arguments", []) or [])
        tau = [float(getattr(s, "tau", 0.0)) for s in strengths]
        decision = pred.decision.value if hasattr(pred.decision, "value") else str(pred.decision)
        gold = getattr(pred, "gold_label", None)
        rows.append({
            "case_id": pred.case_id,
            "domain": domains.get(pred.case_id, ""),
            "gold_label": gold,
            "decision": decision,
            "is_correct": ((decision if decision != "UNCERTAIN" else "No") == gold) if gold in {"Yes", "No"} else None,
            "sigma_phi": float(getattr(pred, "sigma_phi", 0.5)),
            "is_uncertain": bool(getattr(pred.uncertainty, "is_uncertain", False)),
            "claim_text": extraction.claim.text if extraction.claim else "",
            "n_raw_arguments": n_raw,
            "n_kept_arguments": n_kept,
            "n_neutral_arguments": max(0, n_raw - n_kept),
            "n_relations": len(rels),
            "n_phi_edges": sum(1 for r in rels if getattr(r, "target", None) == "phi"),
            "n_strengths": len(strengths),
            "mean_tau": sum(tau) / len(tau) if tau else 0.0,
            "parse_error": extraction.parse_error or "",
        })
    (out_dir / "argument_diagnostics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if rows:
        with (out_dir / "argument_diagnostics.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)


def run_with_solver(full_preds: list[ADSSPrediction], cfg: dict, solver_name: str) -> list[ADSSPrediction]:
    cfg2 = {**cfg, "qbaf": {**cfg.get("qbaf", {}), "solver": solver_name}}
    solver = get_solver(cfg2)
    solver_type = SolverType(solver_name)
    phi_tau = cfg2.get("qbaf", {}).get("phi_initial_strength", 0.5)
    out: list[ADSSPrediction] = []
    for pred in full_preds:
        new = pred.model_copy(deep=True)
        graph = build_qbaf(pred.extraction, pred.strengths, phi_tau, solver_type)
        so = solver.solve(graph, pred.case_id)
        new.solver_output = so
        new.sigma_phi = so.sigma_phi
        new.decision = Decision(make_decision(so.sigma_phi, cfg2)[0])
        out.append(new)
    return out


def run_no_relation_extraction(full_preds: list[ADSSPrediction], cfg: dict) -> list[ADSSPrediction]:
    solver = get_solver(cfg)
    solver_type = SolverType(cfg.get("qbaf", {}).get("solver", "df_quad"))
    phi_tau = cfg.get("qbaf", {}).get("phi_initial_strength", 0.5)
    out: list[ADSSPrediction] = []
    for pred in full_preds:
        new = pred.model_copy(deep=True)
        graph = build_qbaf(pred.extraction, pred.strengths, phi_tau, solver_type)
        graph.edges = [e for e in graph.edges if e.target == "phi"]
        so = solver.solve(graph, pred.case_id)
        new.solver_output = so
        new.sigma_phi = so.sigma_phi
        new.decision = Decision(make_decision(so.sigma_phi, cfg)[0])
        out.append(new)
    return out


def run_no_uae(full_preds: list[ADSSPrediction], cfg: dict) -> list[ADSSPrediction]:
    th = cfg.get("qbaf", {}).get("decision_threshold", 0.5)
    out: list[ADSSPrediction] = []
    for pred in full_preds:
        new = pred.model_copy(deep=True)
        new.decision = Decision.YES if pred.sigma_phi >= th else Decision.NO
        if new.uncertainty:
            new.uncertainty.is_uncertain = False
            new.uncertainty.escalation_triggered = False
        out.append(new)
    return out


def metric_row(m: Any) -> dict[str, Any]:
    return {
        "system_name": m.system_name,
        "n": m.n,
        "n_uncertain": m.n_uncertain,
        "accuracy": m.accuracy.score,
        "accuracy_ci_low": m.accuracy.ci_low,
        "accuracy_ci_high": m.accuracy.ci_high,
        "macro_f1": m.macro_f1.score,
        "macro_f1_ci_low": m.macro_f1.ci_low,
        "macro_f1_ci_high": m.macro_f1.ci_high,
        "false_certainty_rate": m.false_certainty_rate,
    }


def write_ablation(rows: list[dict[str, Any]], out_dir: Path) -> None:
    (out_dir / "ablation_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if rows:
        with (out_dir / "ablation_results.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)


def run_single_suite(suite: str, args: argparse.Namespace, cfg: dict, out_root: Path) -> None:
    suite_name = "hearsay" if suite in ("hearsay", "legalbench") else suite
    out_dir = out_root / suite_name
    out_dir.mkdir(parents=True, exist_ok=True)

    items = suite_items(suite, cfg, args.split, args.max)
    domains = {ex.case_id: domain for ex, _phi, domain in items}
    print(f"Running suite={suite_name} with {len(items)} examples → {out_dir}")

    pipeline = ADSSPipeline(cfg)
    solver = get_solver(cfg)

    print("Running Full ADSS...")
    full_preds = run_full_adss_with_checkpoints(pipeline, items, args, out_dir)

    systems: dict[str, list[Any]] = {"Full ADSS": full_preds}
    non_symbolic = non_symbolic_predictions(full_preds, cfg)

    if not args.skip_baselines:
        fb = None if args.no_fallback else lambda name: (lambda item, e: failed_baseline_prediction(item, e, name))
        print("Running Zero-shot CoT baseline...")
        systems["Zero-shot CoT"] = run_with_delay(
            lambda item: baseline_zero_shot_cot(item, cfg),
            items,
            args.delay,
            fallback_factory=None if args.no_fallback else fb("zero_shot_cot_failed"),
        )
        print("Running Few-shot Prompting baseline...")
        systems["Few-shot Prompting"] = run_with_delay(
            lambda item: baseline_few_shot(item, cfg),
            items,
            args.delay,
            fallback_factory=None if args.no_fallback else fb("few_shot_failed"),
        )

    systems["ADSS w/o Symbolic Solver"] = non_symbolic

    band = tuple(cfg.get("qbaf", {}).get("uncertainty_band", [0.45, 0.55]))
    ordered = [n for n in ["Zero-shot CoT", "Few-shot Prompting", "ADSS w/o Symbolic Solver", "Full ADSS"] if n in systems]

    print("Writing main metrics...")
    metrics = [evaluate_system(systems[n], n, band=band, bootstrap_samples=args.bootstrap_samples, seed=args.seed) for n in ordered]
    write_metrics_report(metrics, out_dir)
    write_argument_diagnostics(full_preds, domains, out_dir)

    print("Writing McNemar tests...")
    mcnemar = [asdict(mcnemar_test(full_preds, systems[n], "Full ADSS", n)) for n in ordered if n != "Full ADSS"]
    (out_dir / "mcnemar_tests.json").write_text(json.dumps(mcnemar, indent=2), encoding="utf-8")

    print("Writing ablations...")
    ablations = {
        "Full ADSS": full_preds,
        "No Symbolic Solver": non_symbolic,
        "No Relation Extraction": run_no_relation_extraction(full_preds, cfg),
        "No HITL": full_preds,
        "No UAE": run_no_uae(full_preds, cfg),
        "Fallback Solver": run_with_solver(full_preds, cfg, "df_quad"),
        "QE Solver": run_with_solver(full_preds, cfg, "qe_semantics"),
    }
    write_ablation([metric_row(evaluate_system(v, k, band=band, bootstrap_samples=args.bootstrap_samples, seed=args.seed)) for k, v in ablations.items()], out_dir)

    if not args.skip_contestability:
        print("Writing contestability report...")
        write_contestability_report([
            run_contestability_simulation(full_preds, solver, cfg, "oracle", seed=args.seed),
            run_contestability_simulation(full_preds, solver, cfg, "confidence", seed=args.seed),
        ], out_dir)

    if not args.skip_robustness:
        print("Writing robustness report...")
        write_robustness_report(run_robustness_suite(full_preds, solver, cfg, seed=args.seed), out_dir)

    print("Writing error analysis...")
    errors, props = analyse_errors(full_preds, band)
    write_error_report(errors, props, out_dir)
    print(f"Done suite={suite_name}. Reports written to: {out_dir}")


def main() -> None:
    args = parse_args()
    cfg = load_config()
    if args.backend:
        cfg["backend"] = args.backend
    cfg.setdefault("qbaf", {})["solver"] = args.solver
    if cfg.get("backend", "gemini").lower() == "lmstudio":
        cfg.setdefault("lmstudio", {})
        cfg["lmstudio"].setdefault("max_tokens", 260000)
        cfg["lmstudio"].setdefault("disable_thinking", True)
    setup_logging(cfg)
    ensure_api_key(cfg)

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    suites = ["hearsay", "contract_nli"] if args.suite == "all_legalbench" else [args.suite]
    for suite in suites:
        run_single_suite(suite, args, cfg, out_root)


if __name__ == "__main__":
    main()
