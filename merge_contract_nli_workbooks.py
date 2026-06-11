import pandas as pd
from pathlib import Path

base = Path("artifacts/evaluation_suite/contract_nli")
paper = base / "contract_nli_paper_tables.xlsx"
audit = base / "contract_nli_false_positive_audit.xlsx"
out = base / "contract_nli_all_results.xlsx"

with pd.ExcelWriter(out, engine="openpyxl") as writer:
    if paper.exists():
        xls = pd.ExcelFile(paper)
        for sheet in xls.sheet_names:
            df = pd.read_excel(paper, sheet_name=sheet, engine="openpyxl")
            safe = sheet[:31]
            df.to_excel(writer, sheet_name=safe, index=False)

    if audit.exists():
        xls = pd.ExcelFile(audit)
        for sheet in xls.sheet_names:
            df = pd.read_excel(audit, sheet_name=sheet, engine="openpyxl")
            safe = ("Audit_" + sheet)[:31]
            df.to_excel(writer, sheet_name=safe, index=False)

print(f"Wrote {out}")
