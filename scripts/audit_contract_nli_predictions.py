#!/usr/bin/env python3
"""Audit Contract NLI predictions for positive bias, timeouts, and false positives.

Run after a Contract NLI evaluation:
python scripts/audit_contract_nli_predictions.py --workbook artifacts/evaluation_suite/contract_nli/contract_nli_paper_tables.xlsx
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

LABELS = ["Yes", "No"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--workbook", default="artifacts/evaluation_suite/contract_nli/contract_nli_paper_tables.xlsx")
    p.add_argument("--out", default="artifacts/evaluation_suite/contract_nli/contract_nli_false_positive_audit.xlsx")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    wb = Path(args.workbook)
    if not wb.exists():
        raise FileNotFoundError(wb)
    diag = pd.read_excel(wb, sheet_name="Diagnostics_Strict", engine="openpyxl")
    for c in ["gold_label", "decision", "parse_error"]:
        if c in diag.columns:
            diag[c] = diag[c].fillna("").astype(str)
    diag["is_timeout"] = diag["parse_error"].str.contains("timed out", case=False, na=False)
    diag["is_false_positive"] = (diag["gold_label"] == "No") & (diag["decision"] == "Yes")
    diag["is_false_negative"] = (diag["gold_label"] == "Yes") & (diag["decision"] == "No")

    summary = pd.DataFrame([{
        "n_total": len(diag),
        "gold_yes": int((diag["gold_label"] == "Yes").sum()),
        "gold_no": int((diag["gold_label"] == "No").sum()),
        "pred_yes": int((diag["decision"] == "Yes").sum()),
        "pred_no": int((diag["decision"] == "No").sum()),
        "pred_uncertain": int((diag["decision"] == "UNCERTAIN").sum()),
        "timeouts": int(diag["is_timeout"].sum()),
        "false_positive_no_as_yes": int(diag["is_false_positive"].sum()),
        "false_negative_yes_as_no": int(diag["is_false_negative"].sum()),
        "no_pred_yes_rate": float(diag.loc[diag["gold_label"] == "No", "decision"].eq("Yes").mean()),
    }])
    conf = pd.crosstab(diag["gold_label"], diag["decision"])
    fp = diag[diag["is_false_positive"]].sort_values("sigma_phi", ascending=False)
    timeouts = diag[diag["is_timeout"]]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        conf.to_excel(writer, sheet_name="confusion")
        fp.to_excel(writer, sheet_name="false_positives", index=False)
        timeouts.to_excel(writer, sheet_name="timeouts", index=False)
    print(f"[OK] Wrote audit workbook: {out}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
