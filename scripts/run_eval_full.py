#!/usr/bin/env python3
"""
Full evaluation script for the research paper.
Runs ALL experiments described in the paper:
  1. Main results      (Table 1): 4 systems
  2. Ablation          (Table 2): 7 variants
  3. Contestability    (Table 3): DFR, CFR, MEC, SSM
  4. Uncertainty eval  (Table 4): certain vs borderline
  5. Robustness        (Figure ): perturbation curves
  6. Error analysis    (Table 5): 4 error categories

Usage:
  python scripts/run_eval_full.py --split test --max 20 --skip-robustness
  python scripts/run_eval_full.py --split test --max 20 --backend lmstudio
"""
from __future__ import annotations

import argparse, csv, json, logging, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.helpers import load_config, setup_logging, ensure_api_key, flatten_report
from src.data.loader import load_examples
from src.data.models import ADSSPrediction, Decision
from src.pipeline.orchestrator import ADSSPipeline, BaselineRunner
from src.qbaf.graph import build_qbaf
from src.qbaf.solver import get_solver, make_decision
from src.evaluation.evaluator import evaluate_predictions, print_report
from src.evaluation.contestability import compute_contestability, print_contestability
from src.evaluation.robustness import run_robustness, print_robustness
from src.evaluation.error_analysis import analyse_errors, print_error_analysis


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split",  default="test")
    p.add_argument("--max",    type=int, help="Limit samples (quick test)")
    p.add_argument("--solver", default="df_quad")
    p.add_argument("--backend", default=None, choices=["gemini", "lmstudio"])
    p.add_argument("--delay",  type=float, default=2.0,
                   help="Seconds to wait between LLM calls (default: 2.0). "
                        "Increase to avoid rate-limiting on free-tier APIs.")
    p.add_argument("--skip-robustness", action="store_true")
    p.add_argument("--experiment", type=int, default=0,
                   help="Run only one experiment (1-7). 0=all.")
    return p.parse_args()


# ── Rate-limited batch runner ─────────────────────────────────────────────────

def run_batch_with_delay(pipeline, examples, delay: float, save: bool = True):
    """Run pipeline one case at a time with a delay between calls."""
    results = []
    for i, ex in enumerate(examples):
        try:
            pred = pipeline.run_case(ex, save_artifacts=save)
        except Exception as e:
            logging.getLogger(__name__).error(f"[{ex.case_id}] Error: {e}")
            from src.data.models import ExtractionResult
            pred = ADSSPrediction(
                case_id=ex.case_id, input_text=ex.text, gold_label=ex.label,
                extraction=ExtractionResult(case_id=ex.case_id, input_text=ex.text,
                                            parse_error=str(e)),
                sigma_phi=0.5, decision=Decision.UNCERTAIN,
            )
        results.append(pred)
        if delay > 0 and i < len(examples) - 1:
            time.sleep(delay)
    return results


# ── Ablation helpers ──────────────────────────────────────────────────────────

def run_no_relation_extraction(adss_preds, cfg):
    from src.data.models import SolverType
    solver  = get_solver(cfg)
    phi_tau = cfg.get("qbaf", {}).get("phi_initial_strength", 0.5)
    stype   = SolverType(cfg.get("qbaf", {}).get("solver", "df_quad"))
    results = []
    for pred in adss_preds:
        new = pred.model_copy(deep=True)
        graph = build_qbaf(pred.extraction, pred.strengths, phi_tau, stype)
        graph.edges = [e for e in graph.edges if e.target == "phi"]
        solver_out  = solver.solve(graph, pred.case_id)
        sigma_phi   = solver_out.sigma_phi
        dec_str, _  = make_decision(sigma_phi, cfg)
        new.solver_output = solver_out
        new.sigma_phi     = sigma_phi
        new.decision      = Decision(dec_str)
        results.append(new)
    return results


def run_no_uae(adss_preds, cfg):
    threshold = cfg.get("qbaf", {}).get("decision_threshold", 0.5)
    results = []
    for pred in adss_preds:
        new = pred.model_copy(deep=True)
        new.decision = Decision.YES if pred.sigma_phi >= threshold else Decision.NO
        new.uncertainty.is_uncertain = False
        results.append(new)
    return results


def run_with_solver(adss_preds, cfg, solver_name):
    from src.data.models import SolverType
    cfg2    = {**cfg, "qbaf": {**cfg.get("qbaf", {}), "solver": solver_name}}
    solver  = get_solver(cfg2)
    phi_tau = cfg2.get("qbaf", {}).get("phi_initial_strength", 0.5)
    stype   = SolverType(solver_name)
    results = []
    for pred in adss_preds:
        new = pred.model_copy(deep=True)
        if pred.solver_output is None:
            results.append(new); continue
        graph      = build_qbaf(pred.extraction, pred.strengths, phi_tau, stype)
        solver_out = solver.solve(graph, pred.case_id)
        sigma_phi  = solver_out.sigma_phi
        dec_str, _ = make_decision(sigma_phi, cfg2)
        new.solver_output = solver_out
        new.sigma_phi     = sigma_phi
        new.decision      = Decision(dec_str)
        results.append(new)
    return results


def uncertainty_partition(predictions, cfg):
    low, high = cfg.get("qbaf", {}).get("uncertainty_band", [0.45, 0.55])
    certain    = [p for p in predictions if p.gold_label and not (low <= p.sigma_phi <= high)]
    borderline = [p for p in predictions if p.gold_label and (low <= p.sigma_phi <= high)]

    def acc(preds):
        if not preds: return 0.0
        return sum(
            (p.decision.value if p.decision != Decision.UNCERTAIN else "No") == p.gold_label
            for p in preds
        ) / len(preds)

    wrong_certain = [p for p in certain
                     if (p.decision.value if p.decision != Decision.UNCERTAIN else "No")
                     != p.gold_label]
    return {
        "certain_accuracy":     acc(certain),
        "certain_share":        len(certain) / max(1, len(predictions)),
        "borderline_accuracy":  acc(borderline),
        "borderline_share":     len(borderline) / max(1, len(predictions)),
        "post_escalation_acc":  acc(borderline),
        "false_certainty_rate": len(wrong_certain) / max(1, len(certain)),
        "n_certain":            len(certain),
        "n_borderline":         len(borderline),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg  = load_config()
    if args.max:
        cfg.setdefault("dataset", {})["max_samples"] = args.max
    cfg.setdefault("qbaf", {})["solver"] = args.solver
    if args.backend:
        cfg["backend"] = args.backend
    setup_logging(cfg)
    ensure_api_key(cfg)

    logger  = logging.getLogger(__name__)
    out_dir = Path(cfg.get("evaluation", {}).get("output_dir", "artifacts/evaluation"))
    out_dir.mkdir(parents=True, exist_ok=True)
    delay   = args.delay

    backend = cfg.get("backend", "gemini")
    if backend == "gemini" and delay < 2.0:
        logger.warning(
            "Gemini free tier allows only ~20 requests/day. "
            "Consider --delay 3 to avoid rate limiting."
        )

    examples = load_examples(cfg.get("dataset", {}), split=args.split)
    logger.info(f"Loaded {len(examples)} examples ({args.split}).")

    run_all = args.experiment == 0
    adss_preds = None  # populated in exp 1

    # ── 1. Full ADSS ──────────────────────────────────────────────────────────
    if run_all or args.experiment == 1:
        logger.info("=" * 60)
        logger.info("EXPERIMENT 1: Full ADSS")
        pipeline   = ADSSPipeline(cfg)
        adss_preds = run_batch_with_delay(pipeline, examples, delay)
        r_full = evaluate_predictions(adss_preds, "full_adss", cfg, out_dir)
        print_report(r_full)

    # Load cached predictions if skipping exp 1
    if adss_preds is None:
        logger.error("No ADSS predictions — run experiment 1 first (omit --experiment flag).")
        return

    # ── 2. Baselines ──────────────────────────────────────────────────────────
    if run_all or args.experiment == 2:
        logger.info("EXPERIMENT 2: Baselines")
        br = BaselineRunner(cfg)

        logger.info("  Zero-shot CoT…")
        cot_preds = []
        for i, ex in enumerate(examples):
            try:
                cot_preds.append(br.run_cot(ex))
            except Exception as e:
                logger.error(f"CoT failed [{ex.case_id}]: {e}")
                from src.data.models import BaselinePrediction
                cot_preds.append(BaselinePrediction(
                    case_id=ex.case_id, baseline_name="zero_shot_cot",
                    decision=Decision.NO, raw_response="", gold_label=ex.label,
                ))
            if delay > 0 and i < len(examples) - 1:
                time.sleep(delay)
        r_cot = evaluate_predictions(cot_preds, "zero_shot_cot", cfg, out_dir)
        print_report(r_cot)

        logger.info("  Few-shot…")
        fs_preds = []
        for i, ex in enumerate(examples):
            try:
                fs_preds.append(br.run_few_shot(ex))
            except Exception as e:
                logger.error(f"Few-shot failed [{ex.case_id}]: {e}")
                from src.data.models import BaselinePrediction
                fs_preds.append(BaselinePrediction(
                    case_id=ex.case_id, baseline_name="few_shot",
                    decision=Decision.NO, raw_response="", gold_label=ex.label,
                ))
            if delay > 0 and i < len(examples) - 1:
                time.sleep(delay)
        r_fs = evaluate_predictions(fs_preds, "few_shot", cfg, out_dir)
        print_report(r_fs)

        logger.info("  ADSS without symbolic solver…")
        no_sym = []
        for p in adss_preds:
            sigma, dec = br.run_adss_no_symbolic(p.extraction, p.strengths)
            no_sym.append(p.model_copy(update={
                "sigma_phi": sigma, "decision": dec, "solver_output": None
            }))
        r_nosym = evaluate_predictions(no_sym, "adss_no_symbolic", cfg, out_dir)
        print_report(r_nosym)

    # ── 3. Ablations ──────────────────────────────────────────────────────────
    if run_all or args.experiment == 3:
        logger.info("EXPERIMENT 3: Ablation study")
        no_rel  = run_no_relation_extraction(adss_preds, cfg)
        r_norel = evaluate_predictions(no_rel, "no_relation_extraction", cfg, out_dir)
        print_report(r_norel)

        no_uae  = run_no_uae(adss_preds, cfg)
        r_nouae = evaluate_predictions(no_uae, "no_uae", cfg, out_dir)
        print_report(r_nouae)

        qe_preds    = run_with_solver(adss_preds, cfg, "qe_semantics")
        r_qe        = evaluate_predictions(qe_preds, "qe_solver", cfg, out_dir)
        print_report(r_qe)

        dfq_preds   = run_with_solver(adss_preds, cfg, "df_quad")
        r_dfquad    = evaluate_predictions(dfq_preds, "dfquad_solver", cfg, out_dir)
        print_report(r_dfquad)

    # ── 4. Contestability ─────────────────────────────────────────────────────
    if run_all or args.experiment == 4:
        logger.info("EXPERIMENT 4: Contestability")
        solver   = get_solver(cfg)
        r_oracle = compute_contestability(adss_preds, solver, cfg, regime="oracle")
        print_contestability(r_oracle)
        r_conf   = compute_contestability(adss_preds, solver, cfg, regime="confidence")
        print_contestability(r_conf)
        for r in [r_oracle, r_conf]:
            with open(out_dir / f"contestability_{r.regime}.json", "w", encoding="utf-8") as f:
                json.dump({"regime": r.regime, "dfr": r.dfr, "cfr": r.cfr,
                           "mec": r.mec, "ssm": r.ssm,
                           "n_total": r.n_total, "details": r.details}, f, indent=2)

    # ── 5. Uncertainty ────────────────────────────────────────────────────────
    if run_all or args.experiment == 5:
        logger.info("EXPERIMENT 5: Uncertainty evaluation")
        uae_stats = uncertainty_partition(adss_preds, cfg)
        print("\n  Uncertainty-Aware Evaluation")
        for k, v in uae_stats.items():
            print(f"    {k:35s}: {v:.4f}")
        with open(out_dir / "uncertainty_eval.json", "w", encoding="utf-8") as f:
            json.dump(uae_stats, f, indent=2)

    # ── 6. Robustness ─────────────────────────────────────────────────────────
    if (run_all or args.experiment == 6) and not args.skip_robustness:
        logger.info("EXPERIMENT 6: Robustness")
        solver  = get_solver(cfg)
        all_rob = []
        for ptype in ["arg_removal", "relation_flip", "tau_noise", "low_conf_pruning"]:
            rob = run_robustness(adss_preds, solver, cfg, perturbation=ptype)
            print_robustness(rob)
            all_rob.extend([{"perturbation": r.perturbation, "p": r.p_level,
                             "accuracy": r.accuracy, "macro_f1": r.macro_f1,
                             "score_shift": r.mean_score_shift} for r in rob])
        with open(out_dir / "robustness.json", "w", encoding="utf-8") as f:
            json.dump(all_rob, f, indent=2)

    # ── 7. Error analysis ─────────────────────────────────────────────────────
    if run_all or args.experiment == 7:
        logger.info("EXPERIMENT 7: Error analysis")
        props = analyse_errors(adss_preds, out_dir)
        print_error_analysis(props)

    # ── Summary CSV ───────────────────────────────────────────────────────────
    if run_all:
        rows = []
        for name in ["full_adss", "zero_shot_cot", "few_shot", "adss_no_symbolic",
                     "no_relation_extraction", "no_uae", "qe_solver", "dfquad_solver"]:
            rfile = out_dir / f"{name}_report.json"
            if rfile.exists():
                d = json.loads(rfile.read_text(encoding="utf-8"))
                rows.append({"system": name,
                             "accuracy": d["accuracy"], "macro_f1": d["macro_f1"],
                             "n": d["n_samples"]})
        if rows:
            with open(out_dir / "summary_all.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader(); w.writerows(rows)

        # Contestability summary
        c_rows = []
        for regime in ["oracle", "confidence"]:
            cf = out_dir / f"contestability_{regime}.json"
            if cf.exists():
                d = json.loads(cf.read_text(encoding="utf-8"))
                c_rows.append({k: d[k] for k in ["regime","dfr","cfr","mec","ssm","n_total"]})
        if c_rows:
            with open(out_dir / "contestability_summary.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(c_rows[0].keys()))
                w.writeheader(); w.writerows(c_rows)

    logger.info(f"\nAll results saved to {out_dir}/")


if __name__ == "__main__":
    main()
