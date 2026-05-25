#!/usr/bin/env python3
"""Build one Excel workbook with all paper-ready Hearsay results.

Run from the project root after `run_evaluation_suite.py` and `make_strict_reports.py`:

python scripts/make_paper_workbook.py --results-dir artifacts/evaluation_suite/hearsay --out artifacts/evaluation_suite/hearsay/hearsay_paper_tables.xlsx

Required/expected files in --results-dir:
- main_results.csv
- ablation_results.csv
- contestability_metrics.csv
- robustness_results.csv
- main_results_strict.csv
- uncertainty_results_strict.csv
- argument_diagnostics.csv or argument_diagnostics_strict.csv
- error_analysis.json
- error_cases.csv
- mcnemar_tests.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="artifacts/evaluation_suite/hearsay")
    p.add_argument("--out", default=None)
    return p.parse_args()


def safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"[WARN] Missing CSV: {path}")
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"[WARN] Could not read {path}: {exc}")
        return None


def safe_read_json(path: Path) -> Any | None:
    if not path.exists():
        print(f"[WARN] Missing JSON: {path}")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[WARN] Could not read {path}: {exc}")
        return None


def pct_df(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c + "_pct"] = (pd.to_numeric(out[c], errors="coerce") * 100).round(2)
    return out


def summarize_main(df: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "system_name", "n", "n_uncertain", "accuracy", "macro_f1",
        "false_certainty_rate", "certain_n", "certain_accuracy", "certain_macro_f1",
        "borderline_n", "borderline_accuracy", "borderline_macro_f1",
    ]
    cols = [c for c in keep if c in df.columns]
    out = df[cols].copy()
    for c in ["accuracy", "macro_f1", "false_certainty_rate", "certain_accuracy", "certain_macro_f1", "borderline_accuracy", "borderline_macro_f1"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").round(4)
            out[c + "_%"] = (out[c] * 100).round(2)
    return out


def summarize_ablation(df: pd.DataFrame) -> pd.DataFrame:
    keep = ["system_name", "n", "n_uncertain", "accuracy", "macro_f1", "false_certainty_rate"]
    cols = [c for c in keep if c in df.columns]
    out = df[cols].copy()
    for c in ["accuracy", "macro_f1", "false_certainty_rate"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").round(4)
            out[c + "_%"] = (out[c] * 100).round(2)
    return out


def error_analysis_to_df(data: Any) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    if not isinstance(data, dict):
        return None, None
    props = data.get("proportions", {})
    errors = data.get("errors", [])
    prop_df = None
    if props:
        prop_df = pd.DataFrame([
            {"category": k, "proportion": v, "percentage": round(float(v) * 100, 2)}
            for k, v in props.items()
        ])
    err_df = pd.DataFrame(errors) if errors else None
    return prop_df, err_df


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    out_path = Path(args.out) if args.out else results_dir / "hearsay_paper_tables.xlsx"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    files = {
        "main": results_dir / "main_results.csv",
        "ablation": results_dir / "ablation_results.csv",
        "contestability": results_dir / "contestability_metrics.csv",
        "robustness": results_dir / "robustness_results.csv",
        "strict": results_dir / "main_results_strict.csv",
        "uncertainty": results_dir / "uncertainty_results_strict.csv",
        "diagnostics_strict": results_dir / "argument_diagnostics_strict.csv",
        "diagnostics": results_dir / "argument_diagnostics.csv",
        "error_cases": results_dir / "error_cases.csv",
    }

    dataframes: dict[str, pd.DataFrame] = {}
    for name, path in files.items():
        df = safe_read_csv(path)
        if df is not None:
            dataframes[name] = df

    error_json = safe_read_json(results_dir / "error_analysis.json")
    error_props_df, error_json_cases_df = error_analysis_to_df(error_json)

    mcnemar_json = safe_read_json(results_dir / "mcnemar_tests.json")
    mcnemar_df = pd.DataFrame(mcnemar_json) if isinstance(mcnemar_json, list) else None

    # Build paper-ready summaries
    sheets: dict[str, pd.DataFrame] = {}
    if "main" in dataframes:
        sheets["Table1_Main"] = summarize_main(dataframes["main"])
        sheets["Raw_Main"] = dataframes["main"]
    if "ablation" in dataframes:
        sheets["Table3_Ablation"] = summarize_ablation(dataframes["ablation"])
        sheets["Raw_Ablation"] = dataframes["ablation"]
    if "contestability" in dataframes:
        c = dataframes["contestability"].copy()
        for col in ["decision_flip_rate", "corrective_flip_rate", "score_shift_magnitude"]:
            if col in c.columns:
                c[col] = pd.to_numeric(c[col], errors="coerce").round(4)
                if "rate" in col:
                    c[col + "_%"] = (c[col] * 100).round(2)
        sheets["Table4_Contestability"] = c
    if "strict" in dataframes:
        s = dataframes["strict"].copy()
        for col in [
            "abstention_rate", "coverage", "accuracy_strict_abstention_wrong",
            "macro_f1_strict_abstention_wrong", "selective_accuracy_on_non_abstained",
            "selective_macro_f1_on_non_abstained", "false_certainty_rate_all_cases",
            "false_certainty_rate_certain_cases",
        ]:
            if col in s.columns:
                s[col] = pd.to_numeric(s[col], errors="coerce").round(4)
                s[col + "_%"] = (s[col] * 100).round(2)
        sheets["Table5_Strict"] = s
    if "uncertainty" in dataframes:
        u = dataframes["uncertainty"].copy()
        for col in ["share_of_cases", "accuracy", "macro_f1"]:
            if col in u.columns:
                numeric = pd.to_numeric(u[col], errors="coerce")
                u[col + "_%"] = (numeric * 100).round(2)
        sheets["Table5_Uncertainty"] = u
    if "robustness" in dataframes:
        r = dataframes["robustness"].copy()
        for col in ["score_stability_delta_p", "accuracy", "macro_f1"]:
            if col in r.columns:
                r[col] = pd.to_numeric(r[col], errors="coerce").round(4)
        sheets["Robustness"] = r
    if error_props_df is not None:
        sheets["Table6_ErrorProps"] = error_props_df
    if "error_cases" in dataframes:
        sheets["Error_Cases"] = dataframes["error_cases"]
    elif error_json_cases_df is not None:
        sheets["Error_Cases"] = error_json_cases_df
    if mcnemar_df is not None:
        sheets["McNemar"] = mcnemar_df
    if "diagnostics_strict" in dataframes:
        sheets["Diagnostics_Strict"] = dataframes["diagnostics_strict"]
    elif "diagnostics" in dataframes:
        sheets["Diagnostics"] = dataframes["diagnostics"]

    # Index sheet
    index_rows = [
        {"sheet": "Table1_Main", "paper_use": "Main benchmark comparison: Zero-shot, Few-shot, ADSS variants"},
        {"sheet": "Table3_Ablation", "paper_use": "Ablation study"},
        {"sheet": "Table4_Contestability", "paper_use": "Contestability simulation"},
        {"sheet": "Table5_Strict", "paper_use": "Strict abstention-aware summary"},
        {"sheet": "Table5_Uncertainty", "paper_use": "Coverage/selective accuracy/abstention table"},
        {"sheet": "Robustness", "paper_use": "Robustness under argument removal, relation flips, tau noise"},
        {"sheet": "Table6_ErrorProps", "paper_use": "Error-category proportions"},
        {"sheet": "Error_Cases", "paper_use": "Misclassified / abstained error cases"},
        {"sheet": "McNemar", "paper_use": "Pairwise significance tests"},
        {"sheet": "Diagnostics_Strict", "paper_use": "Appendix diagnostics and case-level audit"},
    ]
    sheets = {"README_Index": pd.DataFrame(index_rows), **sheets}

    try:
        import openpyxl  # noqa: F401
    except ModuleNotFoundError:
        print("[ERROR] openpyxl is required for .xlsx creation. Install with: pip install openpyxl")
        print("[INFO] CSV files are already available in the results directory.")
        return

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            # Excel sheet names max length 31
            safe_name = sheet_name[:31]
            df.to_excel(writer, sheet_name=safe_name, index=False)
            ws = writer.book[safe_name]
            ws.freeze_panes = "A2"
            # simple width adjustment
            for col_cells in ws.columns:
                max_len = 0
                col_letter = col_cells[0].column_letter
                for cell in col_cells[:100]:
                    try:
                        max_len = max(max_len, len(str(cell.value)) if cell.value is not None else 0)
                    except Exception:
                        pass
                ws.column_dimensions[col_letter].width = min(max(max_len + 2, 10), 55)

    print(f"[OK] Wrote paper workbook: {out_path}")
    print("[OK] Sheets:", ", ".join(sheets.keys()))


if __name__ == "__main__":
    main()
