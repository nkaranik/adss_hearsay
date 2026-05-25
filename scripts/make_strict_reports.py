#!/usr/bin/env python3
"""Create strict abstention-aware and Excel/CSV-friendly evaluation reports.

This post-processing script reads full_adss_predictions.json and writes strict
metrics where UNCERTAIN is treated as abstention, not as No.

It does NOT require openpyxl. If openpyxl is installed, it also writes an .xlsx
workbook. Otherwise it writes CSV files only and continues normally.

Example:
python scripts/make_strict_reports.py --predictions artifacts/evaluation_suite/hearsay/full_adss_predictions.json --out artifacts/evaluation_suite/hearsay
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import pandas as pd

LABELS = ["Yes", "No"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate strict abstention-aware ADSS reports.")
    p.add_argument("--predictions", required=True, help="Path to full_adss_predictions.json")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--band-low", type=float, default=0.45)
    p.add_argument("--band-high", type=float, default=0.55)
    p.add_argument("--system-name", default="Full ADSS")
    p.add_argument("--no-xlsx", action="store_true", help="Do not attempt to write Excel workbook.")
    return p.parse_args()


def decision_value(x: Any) -> str:
    if isinstance(x, dict):
        return str(x.get("value") or x.get("decision") or x)
    return str(x)


def binary_metrics(golds: list[str], decisions: list[str]) -> tuple[float, float, dict[str, dict[str, float]]]:
    """Accuracy and Macro-F1 over Yes/No, with UNCERTAIN counted as abstention/wrong.

    For Macro-F1, an abstention counts as a false negative for the true class and
    does not count as a false positive for the opposite class.
    """
    n = len(golds)
    accuracy = sum(g == d for g, d in zip(golds, decisions)) / n if n else 0.0
    per_class: dict[str, dict[str, float]] = {}
    f1s: list[float] = []
    for label in LABELS:
        tp = sum(g == label and d == label for g, d in zip(golds, decisions))
        fp = sum(g != label and d == label for g, d in zip(golds, decisions))
        fn = sum(g == label and d != label for g, d in zip(golds, decisions))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
        f1s.append(f1)
    return accuracy, sum(f1s) / len(f1s), per_class


def prediction_rows(preds: list[dict[str, Any]], band_low: float, band_high: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for p in preds:
        extraction = p.get("extraction") or {}
        args = extraction.get("arguments") or []
        rels = extraction.get("relations") or []
        strengths = p.get("strengths") or []
        tau_vals = [float(s.get("tau", 0.0)) for s in strengths]
        sigma = float(p.get("sigma_phi", 0.5))
        decision = decision_value(p.get("decision"))
        uncertainty = p.get("uncertainty") or {}
        is_abstention = decision == "UNCERTAIN" or bool(uncertainty.get("is_uncertain", False))
        rows.append({
            "case_id": p.get("case_id"),
            "gold_label": p.get("gold_label"),
            "decision": decision,
            "is_abstention": is_abstention,
            "is_correct_strict": decision == p.get("gold_label"),
            "sigma_phi": sigma,
            "sigma_phi_fixed6": f"{sigma:.6f}",
            "sigma_phi_fixed4": f"{sigma:.4f}",
            "sigma_phi_percent": sigma * 100.0,
            "confidence_margin_from_0_5": abs(sigma - 0.5),
            "in_uncertainty_band": band_low <= sigma <= band_high,
            "n_arguments": len(args),
            "n_relations": len(rels),
            "n_phi_edges": sum(1 for r in rels if r.get("target") == "phi"),
            "n_strengths": len(strengths),
            "mean_tau": sum(tau_vals) / len(tau_vals) if tau_vals else 0.0,
            "parse_error": extraction.get("parse_error") or "",
        })
    return pd.DataFrame(rows)


def write_excel_if_possible(
    xlsx: Path,
    summary: dict[str, Any],
    uncertainty_rows: list[dict[str, Any]],
    conf: pd.DataFrame,
    case_df: pd.DataFrame,
    per_class: dict[str, dict[str, float]],
    sel_per_class: dict[str, dict[str, float]],
) -> None:
    try:
        import openpyxl  # noqa: F401
    except ModuleNotFoundError:
        print("[INFO] openpyxl is not installed; skipped .xlsx creation. CSV files were written successfully.")
        print("[INFO] If you want .xlsx locally, run: pip install openpyxl")
        return

    case_display = case_df.copy()
    for col in ["sigma_phi", "sigma_phi_percent", "confidence_margin_from_0_5", "mean_tau"]:
        case_display[col] = case_display[col].round(6)
    summary_display = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in summary.items()}
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        pd.DataFrame([summary_display]).to_excel(writer, sheet_name="strict_summary", index=False)
        pd.DataFrame(uncertainty_rows).to_excel(writer, sheet_name="uncertainty_table", index=False)
        conf.to_excel(writer, sheet_name="confusion")
        case_display.to_excel(writer, sheet_name="case_diagnostics", index=False)
        pd.DataFrame(per_class).T.to_excel(writer, sheet_name="per_class_strict")
        if sel_per_class:
            pd.DataFrame(sel_per_class).T.to_excel(writer, sheet_name="per_class_selective")
    print(f"[OK] Wrote Excel workbook: {xlsx}")


def main() -> None:
    args = parse_args()
    predictions_path = Path(args.predictions)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    preds = json.loads(predictions_path.read_text(encoding="utf-8"))

    case_df = prediction_rows(preds, args.band_low, args.band_high)
    golds = case_df["gold_label"].tolist()
    decisions = case_df["decision"].tolist()
    acc_strict, macro_f1_strict, per_class = binary_metrics(golds, decisions)

    certain_df = case_df[~case_df["is_abstention"]].copy()
    abstain_df = case_df[case_df["is_abstention"]].copy()
    if len(certain_df):
        sel_acc, sel_macro_f1, sel_per_class = binary_metrics(
            certain_df["gold_label"].tolist(), certain_df["decision"].tolist()
        )
    else:
        sel_acc = sel_macro_f1 = 0.0
        sel_per_class = {}

    false_certain = len(certain_df[certain_df["gold_label"] != certain_df["decision"]])
    summary = {
        "system_name": args.system_name,
        "n": len(case_df),
        "n_yes_gold": int((case_df["gold_label"] == "Yes").sum()),
        "n_no_gold": int((case_df["gold_label"] == "No").sum()),
        "n_abstentions": int(case_df["is_abstention"].sum()),
        "abstention_rate": float(case_df["is_abstention"].mean()),
        "coverage": float(len(certain_df) / len(case_df)) if len(case_df) else 0.0,
        "accuracy_strict_abstention_wrong": acc_strict,
        "macro_f1_strict_abstention_wrong": macro_f1_strict,
        "selective_accuracy_on_non_abstained": sel_acc,
        "selective_macro_f1_on_non_abstained": sel_macro_f1,
        "false_certainty_count": int(false_certain),
        "false_certainty_rate_all_cases": float(false_certain / len(case_df)) if len(case_df) else 0.0,
        "false_certainty_rate_certain_cases": float(false_certain / len(certain_df)) if len(certain_df) else 0.0,
        "mean_sigma_phi": float(case_df["sigma_phi"].mean()),
        "min_sigma_phi": float(case_df["sigma_phi"].min()),
        "max_sigma_phi": float(case_df["sigma_phi"].max()),
        "mean_arguments": float(case_df["n_arguments"].mean()),
        "median_arguments": float(case_df["n_arguments"].median()),
        "cases_3plus_arguments": int((case_df["n_arguments"] >= 3).sum()),
        "cases_4plus_arguments": int((case_df["n_arguments"] >= 4).sum()),
    }

    uncertainty_rows = [
        {
            "category": "Certain / non-abstained cases",
            "n": len(certain_df),
            "share_of_cases": len(certain_df) / len(case_df) if len(case_df) else 0.0,
            "accuracy": sel_acc,
            "macro_f1": sel_macro_f1,
        },
        {
            "category": "Borderline / abstained cases",
            "n": len(abstain_df),
            "share_of_cases": len(abstain_df) / len(case_df) if len(case_df) else 0.0,
            "accuracy": "N/A - abstained",
            "macro_f1": "N/A - abstained",
        },
        {
            "category": "All cases, abstention counted wrong",
            "n": len(case_df),
            "share_of_cases": 1.0,
            "accuracy": acc_strict,
            "macro_f1": macro_f1_strict,
        },
    ]

    conf = pd.crosstab(case_df["gold_label"], case_df["decision"])

    # CSV outputs. fixed6/fixed4 columns prevent Excel scientific-notation confusion.
    case_df.to_csv(out_dir / "argument_diagnostics_strict.csv", index=False, float_format="%.10f")
    pd.DataFrame([summary]).to_csv(out_dir / "main_results_strict.csv", index=False, float_format="%.10f")
    pd.DataFrame(uncertainty_rows).to_csv(out_dir / "uncertainty_results_strict.csv", index=False, float_format="%.10f")
    conf.to_csv(out_dir / "confusion_with_abstention.csv")
    pd.DataFrame(per_class).T.to_csv(out_dir / "per_class_strict.csv", float_format="%.10f")
    if sel_per_class:
        pd.DataFrame(sel_per_class).T.to_csv(out_dir / "per_class_selective.csv", float_format="%.10f")

    summary_json = {
        "summary": summary,
        "uncertainty_rows": uncertainty_rows,
        "per_class_strict": per_class,
        "per_class_selective": sel_per_class,
    }
    (out_dir / "strict_report_summary.json").write_text(json.dumps(summary_json, indent=2), encoding="utf-8")

    if not args.no_xlsx:
        write_excel_if_possible(
            out_dir / "strict_abstention_excel_friendly.xlsx",
            summary,
            uncertainty_rows,
            conf,
            case_df,
            per_class,
            sel_per_class,
        )

    print(json.dumps(summary, indent=2))
    print(f"[OK] Wrote strict CSV/JSON reports to: {out_dir}")


if __name__ == "__main__":
    main()
