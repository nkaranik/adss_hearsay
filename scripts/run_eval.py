#!/usr/bin/env python3
"""
Reproducible evaluation script (Gemini backend).
Usage:  python scripts/run_eval.py [--split test] [--max 20]
"""
from __future__ import annotations

import argparse, csv, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.helpers import load_config, setup_logging, ensure_api_key, flatten_report
from src.data.loader import load_examples
from src.pipeline.orchestrator import ADSSPipeline, BaselineRunner
from src.evaluation.evaluator import evaluate_predictions, print_report


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split",  default="test")
    p.add_argument("--max",    type=int)
    p.add_argument("--solver", default="df_quad")
    p.add_argument("--baselines-only", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = load_config()
    if args.max:
        cfg.setdefault("dataset", {})["max_samples"] = args.max
    cfg.setdefault("qbaf", {})["solver"] = args.solver
    setup_logging(cfg)
    ensure_api_key()

    import logging
    logger  = logging.getLogger(__name__)
    out_dir = Path(cfg.get("evaluation", {}).get("output_dir", "artifacts/evaluation"))

    examples = load_examples(cfg.get("dataset", {}), split=args.split)
    logger.info(f"Loaded {len(examples)} examples ({args.split}).")

    all_reports = []

    if not args.baselines_only:
        logger.info("Running Full ADSS…")
        pipeline    = ADSSPipeline(cfg)
        adss_preds  = pipeline.run_batch(examples)
        r = evaluate_predictions(adss_preds, "full_adss", cfg, out_dir)
        print_report(r)
        all_reports.append(flatten_report(r))

        logger.info("Running ADSS-no-symbolic…")
        br = BaselineRunner(cfg)
        no_sym = []
        for p in adss_preds:
            sigma, dec = br.run_adss_no_symbolic(p.extraction, p.strengths)
            no_sym.append(p.model_copy(update={
                "sigma_phi": sigma, "decision": dec, "solver_output": None
            }))
        r2 = evaluate_predictions(no_sym, "adss_no_symbolic", cfg, out_dir)
        print_report(r2)
        all_reports.append(flatten_report(r2))

    br = BaselineRunner(cfg)

    logger.info("Running Zero-shot CoT…")
    cot = [br.run_cot(ex) for ex in examples]
    r3  = evaluate_predictions(cot, "zero_shot_cot", cfg, out_dir)
    print_report(r3)
    all_reports.append(flatten_report(r3))

    logger.info("Running Few-shot…")
    fs  = [br.run_few_shot(ex) for ex in examples]
    r4  = evaluate_predictions(fs, "few_shot", cfg, out_dir)
    print_report(r4)
    all_reports.append(flatten_report(r4))

    if all_reports:
        out_dir.mkdir(parents=True, exist_ok=True)
        summary = out_dir / "summary.csv"
        with open(summary, "w", encoding="utf-8", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(all_reports[0].keys()))
            w.writeheader(); w.writerows(all_reports)
        logger.info(f"Summary → {summary}")


if __name__ == "__main__":
    main()
