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
  python scripts/run_eval_full.py --split test --max 100
"""
from __future__ import annotations

import argparse, csv, json, logging, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.helpers import load_config, setup_logging, ensure_api_key, flatten_report
from src.data.loader import load_examples
from src.data.models import ADSSPrediction, Decision
from src.pipeline.orchestrator import ADSSPipeline, BaselineRunner
from src.qbaf.graph import build_qbaf
from src.qbaf.solver import get_solver, DFQuADSolver, QESemanticsSolver, make_decision
from src.evaluation.evaluator import evaluate_predictions, print_report
from src.evaluation.contestability import compute_contestability, print_contestability
from src.evaluation.robustness import run_robustness, print_robustness
from src.evaluation.error_analysis import analyse_errors, print_error_analysis


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split",  default="test")
    p.add_argument("--max",    type=int, help="Limit samples (quick test)")
    p.add_argument("--solver", default="df_quad")
    p.add_argument("--skip-robustness", action="store_true",
                   help="Skip slow robustness experiments")
    p.add_argument("--backend", default=None,
                   choices=["gemini", "lmstudio"],
                   help="Override backend (default: from config.yaml)")
    return p.parse_args()


# ── Ablation helpers ──────────────────────────────────────────────────────────

def run_no_relation_extraction(
    adss_preds: list[ADSSPrediction], cfg: dict
) -> list[ADSSPrediction]:
    """
    Ablation: ignore inter-argument relations.
    Each argument connects to phi directly (stance-based), no arg-arg edges.
    """
    import copy
    from src.data.models import QBAFEdge, RelationType, SolverType

    results = []
    solver  = get_solver(cfg)
    phi_tau = cfg.get("qbaf", {}).get("phi_initial_strength", 0.5)

    for pred in adss_preds:
        new_pred = pred.model_copy(deep=True)
        # Build graph with ONLY arg→phi edges (drop all arg-arg edges)
        graph = build_qbaf(
            pred.extraction, pred.strengths, phi_tau,
            SolverType(cfg.get("qbaf", {}).get("solver", "df_quad")),
        )
        # Remove all edges that don't target phi
        graph.edges = [e for e in graph.edges if e.target == "phi"]

        solver_out  = solver.solve(graph, pred.case_id)
        sigma_phi   = solver_out.sigma_phi
        dec_str, _  = make_decision(sigma_phi, cfg)

        new_pred.solver_output = solver_out
        new_pred.sigma_phi     = sigma_phi
        new_pred.decision      = Decision(dec_str)
        results.append(new_pred)
    return results


def run_no_uae(adss_preds: list[ADSSPrediction], cfg: dict) -> list[ADSSPrediction]:
    """
    Ablation: disable UAE — force binary decision even in uncertainty band.
    """
    threshold = cfg.get("qbaf", {}).get("decision_threshold", 0.5)
    results = []
    for pred in adss_preds:
        new_pred = pred.model_copy(deep=True)
        # Ignore uncertainty band — use threshold only
        dec = Decision.YES if pred.sigma_phi >= threshold else Decision.NO
        new_pred.decision = dec
        new_pred.uncertainty.is_uncertain = False
        results.append(new_pred)
    return results


def run_with_solver(
    adss_preds: list[ADSSPrediction], cfg: dict, solver_name: str
) -> list[ADSSPrediction]:
    """Re-solve all predictions with a different solver."""
    cfg2 = {**cfg, "qbaf": {**cfg.get("qbaf", {}), "solver": solver_name}}
    solver   = get_solver(cfg2)
    phi_tau  = cfg2.get("qbaf", {}).get("phi_initial_strength", 0.5)
    results  = []
    from src.data.models import SolverType
    stype = SolverType(solver_name)

    for pred in adss_preds:
        new_pred = pred.model_copy(deep=True)
        if pred.solver_output is None:
            results.append(new_pred)
            continue
        # Rebuild graph with original strengths and re-solve
        graph       = build_qbaf(pred.extraction, pred.strengths, phi_tau, stype)
        solver_out  = solver.solve(graph, pred.case_id)
        sigma_phi   = solver_out.sigma_phi
        dec_str, _  = make_decision(sigma_phi, cfg2)
        new_pred.solver_output = solver_out
        new_pred.sigma_phi     = sigma_phi
        new_pred.decision      = Decision(dec_str)
        results.append(new_pred)
    return results


# ── Uncertainty partition ─────────────────────────────────────────────────────

def uncertainty_partition(
    predictions: list[ADSSPrediction], cfg: dict
) -> dict:
    low, high = cfg.get("qbaf", {}).get("uncertainty_band", [0.45, 0.55])
    certain   = [p for p in predictions
                 if p.gold_label and not (low <= p.sigma_phi <= high)]
    borderline = [p for p in predictions
                  if p.gold_label and (low <= p.sigma_phi <= high)]

    def acc(preds):
        if not preds:
            return 0.0
        correct = sum(
            (p.decision.value if p.decision != Decision.UNCERTAIN else "No") == p.gold_label
            for p in preds
        )
        return correct / len(preds)

    # Post-escalation: for borderline cases, if we "escalate" (abstain → No as default)
    # measure how often the gold label was also No (conservative estimate)
    post_esc_acc = acc(borderline)   # same as borderline acc in current implementation

    # False certainty rate: wrong predictions OUTSIDE uncertainty band
    wrong_certain = [p for p in certain
                     if (p.decision.value if p.decision != Decision.UNCERTAIN else "No")
                     != p.gold_label]
    false_certainty_rate = len(wrong_certain) / len(certain) if certain else 0.0

    return {
        "certain_accuracy":        acc(certain),
        "certain_share":           len(certain) / max(1, len(predictions)),
        "borderline_accuracy":     acc(borderline),
        "borderline_share":        len(borderline) / max(1, len(predictions)),
        "post_escalation_acc":     post_esc_acc,
        "false_certainty_rate":    false_certainty_rate,
        "n_certain":               len(certain),
        "n_borderline":            len(borderline),
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

    examples = load_examples(cfg.get("dataset", {}), split=args.split)
    logger.info(f"Loaded {len(examples)} examples ({args.split}).")

    # ── 1. Run Full ADSS ──────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("EXPERIMENT 1: Full ADSS")
    pipeline   = ADSSPipeline(cfg)
    adss_preds = pipeline.run_batch(examples)

    r_full = evaluate_predictions(adss_preds, "full_adss", cfg, out_dir)
    print_report(r_full)

    # ── 2. Baselines ──────────────────────────────────────────────────────────
    logger.info("EXPERIMENT 2: Baselines")
    br = BaselineRunner(cfg)

    logger.info("  Zero-shot CoT…")
    cot_preds = [br.run_cot(ex) for ex in examples]
    r_cot = evaluate_predictions(cot_preds, "zero_shot_cot", cfg, out_dir)
    print_report(r_cot)

    logger.info("  Few-shot…")
    fs_preds = [br.run_few_shot(ex) for ex in examples]
    r_fs = evaluate_predictions(fs_preds, "few_shot", cfg, out_dir)
    print_report(r_fs)

    logger.info("  ADSS without symbolic solver…")
    no_sym_preds = []
    for p in adss_preds:
        sigma, dec = br.run_adss_no_symbolic(p.extraction, p.strengths)
        no_sym_preds.append(p.model_copy(update={
            "sigma_phi": sigma, "decision": dec, "solver_output": None
        }))
    r_nosym = evaluate_predictions(no_sym_preds, "adss_no_symbolic", cfg, out_dir)
    print_report(r_nosym)

    # ── 3. Ablations ──────────────────────────────────────────────────────────
    logger.info("EXPERIMENT 3: Ablation study")

    logger.info("  No relation extraction…")
    no_rel_preds = run_no_relation_extraction(adss_preds, cfg)
    r_norel = evaluate_predictions(no_rel_preds, "no_relation_extraction", cfg, out_dir)
    print_report(r_norel)

    logger.info("  No UAE…")
    no_uae_preds = run_no_uae(adss_preds, cfg)
    r_nouae = evaluate_predictions(no_uae_preds, "no_uae", cfg, out_dir)
    print_report(r_nouae)

    logger.info("  QE semantics solver…")
    qe_preds = run_with_solver(adss_preds, cfg, "qe_semantics")
    r_qe = evaluate_predictions(qe_preds, "qe_solver", cfg, out_dir)
    print_report(r_qe)

    logger.info("  DF-QuAD fallback solver…")
    dfquad_preds = run_with_solver(adss_preds, cfg, "df_quad")
    r_dfquad = evaluate_predictions(dfquad_preds, "dfquad_solver", cfg, out_dir)
    print_report(r_dfquad)

    # ── 4. Contestability ─────────────────────────────────────────────────────
    logger.info("EXPERIMENT 4: Contestability")
    solver = get_solver(cfg)

    r_oracle = compute_contestability(adss_preds, solver, cfg, regime="oracle")
    print_contestability(r_oracle)

    r_conf = compute_contestability(adss_preds, solver, cfg, regime="confidence")
    print_contestability(r_conf)

    # Save contestability details
    for r in [r_oracle, r_conf]:
        with open(out_dir / f"contestability_{r.regime}.json", "w") as f:
            json.dump({
                "regime": r.regime, "dfr": r.dfr, "cfr": r.cfr,
                "mec": r.mec, "ssm": r.ssm,
                "n_total": r.n_total, "details": r.details,
            }, f, indent=2)

    # ── 5. Uncertainty partition ───────────────────────────────────────────────
    logger.info("EXPERIMENT 5: Uncertainty evaluation")
    uae_stats = uncertainty_partition(adss_preds, cfg)
    print("\n  Uncertainty-Aware Evaluation")
    for k, v in uae_stats.items():
        print(f"    {k:35s}: {v:.4f}")
    with open(out_dir / "uncertainty_eval.json", "w") as f:
        json.dump(uae_stats, f, indent=2)

    # ── 6. Robustness ─────────────────────────────────────────────────────────
    if not args.skip_robustness:
        logger.info("EXPERIMENT 6: Robustness (may take a while…)")
        all_rob = []
        for ptype in ["arg_removal", "relation_flip", "tau_noise", "low_conf_pruning"]:
            rob = run_robustness(adss_preds, solver, cfg, perturbation=ptype)
            print_robustness(rob)
            all_rob.extend([
                {"perturbation": r.perturbation, "p": r.p_level,
                 "accuracy": r.accuracy, "macro_f1": r.macro_f1,
                 "score_shift": r.mean_score_shift}
                for r in rob
            ])
        with open(out_dir / "robustness.json", "w") as f:
            json.dump(all_rob, f, indent=2)
    else:
        logger.info("Skipping robustness (--skip-robustness).")

    # ── 7. Error analysis ─────────────────────────────────────────────────────
    logger.info("EXPERIMENT 7: Error analysis")
    error_props = analyse_errors(adss_preds, out_dir)
    print_error_analysis(error_props)

    # ── Summary CSV (for paper tables) ────────────────────────────────────────
    summary_rows = [
        flatten_report(r_cot),
        flatten_report(r_fs),
        flatten_report(r_nosym),
        flatten_report(r_full),
        flatten_report(r_norel),
        flatten_report(r_nouae),
        flatten_report(r_qe),
        flatten_report(r_dfquad),
    ]
    summary_path = out_dir / "summary_all.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    # Contestability summary
    with open(out_dir / "contestability_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["regime","dfr","cfr","mec","ssm","n_total"])
        w.writeheader()
        for r in [r_oracle, r_conf]:
            w.writerow({"regime": r.regime, "dfr": r.dfr, "cfr": r.cfr,
                        "mec": r.mec, "ssm": r.ssm, "n_total": r.n_total})

    logger.info(f"\nAll results saved to {out_dir}/")
    logger.info("Files:")
    for f in sorted(out_dir.glob("*.json")) + sorted(out_dir.glob("*.csv")):
        logger.info(f"  {f.name}")


if __name__ == "__main__":
    main()
