#!/usr/bin/env python3
"""Delete stale Contract NLI predictions/evaluation outputs after cache/prompt fix.

Old predictions must be removed because they were produced with input_text that
missed the target statement. After purging, rerun with --resume safely.

Usage:
python scripts/purge_contract_nli_outputs.py --yes
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pred-dir", default="artifacts/contract_nli_predictions")
    p.add_argument("--eval-dir", default="artifacts/evaluation_suite/contract_nli")
    p.add_argument("--yes", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pred_dir = Path(args.pred_dir)
    eval_dir = Path(args.eval_dir)

    targets = []
    if pred_dir.exists():
        targets.extend(pred_dir.glob("test_*_prediction.json"))
    eval_outputs = [
        "full_adss_predictions.json",
        "main_results.csv",
        "ablation_results.csv",
        "contestability_metrics.csv",
        "robustness_results.csv",
        "error_analysis.json",
        "error_cases.csv",
        "argument_diagnostics.csv",
        "argument_diagnostics_strict.csv",
        "main_results_strict.csv",
        "uncertainty_results_strict.csv",
        "strict_abstention_excel_friendly.xlsx",
        "contract_nli_paper_tables.xlsx",
        "mcnemar_tests.json",
    ]
    for name in eval_outputs:
        p = eval_dir / name
        if p.exists():
            targets.append(p)

    print("Will delete:")
    for t in targets:
        print(" -", t)
    if not targets:
        print("[INFO] Nothing to delete.")
        return
    if not args.yes:
        print("\nRe-run with --yes to actually delete these files.")
        return

    for t in targets:
        try:
            t.unlink()
        except IsADirectoryError:
            shutil.rmtree(t)
    print(f"[OK] Deleted {len(targets)} stale Contract NLI output files.")


if __name__ == "__main__":
    main()
