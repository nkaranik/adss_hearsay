
"""
Run the full ADSS-Hearsay evaluation/testing suite.

Typical use:
    python run_evaluation_suite.py --max 20 --skip-robustness
    python run_evaluation_suite.py --split test --max 100 --out artifacts/evaluation_suite

This script runs:
- Full ADSS
- Zero-shot CoT baseline
- Few-shot baseline
- Non-symbolic ADSS baseline
- Metrics + bootstrap CIs + McNemar tests
- Uncertainty partition + false-certainty rate
- Contestability simulations
- Robustness perturbations
- Structured error analysis
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent))

from src.evaluation.metrics import evaluate_system, mcnemar_test, write_metrics_report
from src.evaluation.simulations import run_contestability_simulation, write_contestability_report
from src.evaluation.robustness import run_robustness_suite, write_robustness_report
from src.evaluation.error_analysis import analyse_errors, write_error_report

from src.utils.helpers import load_config, setup_logging, ensure_api_key
from src.data.loader import load_examples
from src.pipeline.orchestrator import ADSSPipeline, BaselineRunner
from src.data.models import BaselinePrediction, Decision
from src.qbaf.solver import get_solver


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test")
    p.add_argument("--max", type=int, default=None)
    p.add_argument("--out", default="artifacts/evaluation_suite")
    p.add_argument("--delay", type=float, default=1.5)
    p.add_argument("--skip-baselines", action="store_true")
    p.add_argument("--skip-contestability", action="store_true")
    p.add_argument("--skip-robustness", action="store_true")
    p.add_argument("--backend", choices=["gemini", "lmstudio"], default=None)
    p.add_argument("--bootstrap-samples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def run_with_delay(fn, items, delay: float):
    out = []
    for i, item in enumerate(items):
        out.append(fn(item))
        if delay and i < len(items) - 1:
            time.sleep(delay)
    return out


def non_symbolic_predictions(full_preds, baseline_runner):
    preds = []
    for p in full_preds:
        sigma, dec = baseline_runner.run_adss_no_symbolic(p.extraction, p.strengths)
        preds.append(BaselinePrediction(
            case_id=p.case_id,
            baseline_name="non_symbolic_adss",
            decision=dec,
            raw_response=json.dumps({"sigma": sigma}),
            gold_label=p.gold_label,
        ))
    return preds


def main():
    args = parse_args()
    cfg = load_config()
    if args.backend:
        cfg["backend"] = args.backend
    if args.max:
        cfg.setdefault("dataset", {})["max_samples"] = args.max
    setup_logging(cfg)
    ensure_api_key(cfg)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    examples = load_examples(cfg["dataset"], args.split)
    if args.max:
        examples = examples[:args.max]

    pipeline = ADSSPipeline(cfg)
    baseline_runner = BaselineRunner(cfg)
    solver = get_solver(cfg)

    print(f"Running Full ADSS on {len(examples)} examples...")
    full_preds = run_with_delay(lambda ex: pipeline.run_case(ex, save_artifacts=False), examples, args.delay)

    systems = {"Full ADSS": full_preds}

    if not args.skip_baselines:
        print("Running Zero-shot CoT baseline...")
        systems["Zero-shot CoT"] = run_with_delay(baseline_runner.run_cot, examples, args.delay)
        print("Running Few-shot baseline...")
        systems["Few-shot"] = run_with_delay(baseline_runner.run_few_shot, examples, args.delay)

    print("Computing Non-Symbolic ADSS baseline...")
    systems["Non-Symbolic ADSS"] = non_symbolic_predictions(full_preds, baseline_runner)

    print("Computing metrics...")
    metrics = [
        evaluate_system(
            preds,
            name,
            band=tuple(cfg.get("qbaf", {}).get("uncertainty_band", [0.45, 0.55])),
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
        for name, preds in systems.items()
    ]
    write_metrics_report(metrics, out_dir)

    mcnemar_results = []
    for name, preds in systems.items():
        if name == "Full ADSS":
            continue
        mcnemar_results.append(asdict(mcnemar_test(full_preds, preds, "Full ADSS", name)))
    (out_dir / "mcnemar_tests.json").write_text(json.dumps(mcnemar_results, indent=2), encoding="utf-8")

    if not args.skip_contestability:
        print("Running contestability simulations...")
        contestability = [
            run_contestability_simulation(full_preds, solver, cfg, "oracle", seed=args.seed),
            run_contestability_simulation(full_preds, solver, cfg, "confidence", seed=args.seed),
        ]
        write_contestability_report(contestability, out_dir)

    if not args.skip_robustness:
        print("Running robustness suite...")
        robustness = run_robustness_suite(full_preds, solver, cfg, seed=args.seed)
        write_robustness_report(robustness, out_dir)

    print("Running structured error analysis...")
    errors, proportions = analyse_errors(full_preds, tuple(cfg.get("qbaf", {}).get("uncertainty_band", [0.45, 0.55])))
    write_error_report(errors, proportions, out_dir)

    print(f"Done. Reports written to: {out_dir}")


if __name__ == "__main__":
    main()
