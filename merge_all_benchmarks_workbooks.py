import pandas as pd
from pathlib import Path
from openpyxl import load_workbook

# Run this from the project root:
#   python merge_all_benchmarks_workbooks.py

SEARCH_PATHS = [
    Path("."),
    Path("artifacts/evaluation_suite/hearsay"),
    Path("artifacts/evaluation_suite/contract_nli"),
]

def find_file(name: str) -> Path | None:
    for base in SEARCH_PATHS:
        p = base / name
        if p.exists():
            return p
    return None

hearsay = find_file("hearsay_paper_tables.xlsx")
hearsay_strict = find_file("hearsay_strict_abstention_excel_friendly.xlsx")
contract_all = find_file("contract_nli_all_results.xlsx")
contract_paper = find_file("contract_nli_paper_tables.xlsx")
contract_audit = find_file("contract_nli_false_positive_audit.xlsx")

if hearsay is None:
    raise FileNotFoundError(
        "Could not find hearsay_paper_tables.xlsx. Put it in the project root "
        "or artifacts/evaluation_suite/hearsay/."
    )
if contract_all is None and contract_paper is None:
    raise FileNotFoundError(
        "Could not find contract_nli_all_results.xlsx or contract_nli_paper_tables.xlsx. "
        "Put it in the project root or artifacts/evaluation_suite/contract_nli/."
    )

out_dir = Path("artifacts/evaluation_suite")
out_dir.mkdir(parents=True, exist_ok=True)
out = out_dir / "all_benchmarks_paper_tables.xlsx"

def safe_sheet_name(prefix: str, sheet: str, used: set[str]) -> str:
    raw = f"{prefix}_{sheet}"
    replacements = {
        "README_Index": "README",
        "Table": "T",
        "Diagnostics": "Diag",
        "Contestability": "Contest",
        "Uncertainty": "Uncert",
        "Robustness": "Robust",
        "false_positives": "FPs",
        "false_positive": "FP",
    }
    for a, b in replacements.items():
        raw = raw.replace(a, b)
    name = raw[:31]
    base = name
    i = 1
    while name in used:
        suffix = f"_{i}"
        name = base[:31-len(suffix)] + suffix
        i += 1
    used.add(name)
    return name

summary_rows = []
used = set()

def copy_workbook_sheets(writer, benchmark: str, workbook: Path, prefix: str):
    xls = pd.ExcelFile(workbook)
    for sheet in xls.sheet_names:
        df = pd.read_excel(workbook, sheet_name=sheet, engine="openpyxl")
        sname = safe_sheet_name(prefix, sheet, used)
        df.to_excel(writer, sheet_name=sname, index=False)
        summary_rows.append({
            "benchmark": benchmark,
            "source_workbook": str(workbook),
            "source_sheet": sheet,
            "combined_sheet": sname,
            "rows": len(df),
            "columns": len(df.columns),
        })

with pd.ExcelWriter(out, engine="openpyxl") as writer:
    copy_workbook_sheets(writer, "hearsay", hearsay, "Hearsay")

    if hearsay_strict is not None:
        copy_workbook_sheets(writer, "hearsay_strict", hearsay_strict, "HStrict")

    if contract_all is not None:
        copy_workbook_sheets(writer, "contract_nli", contract_all, "Contract")
    else:
        copy_workbook_sheets(writer, "contract_nli", contract_paper, "Contract")
        if contract_audit is not None:
            copy_workbook_sheets(writer, "contract_nli_audit", contract_audit, "ContractAudit")

    pd.DataFrame(summary_rows).to_excel(writer, sheet_name="README_Index", index=False)

# Move README_Index first
wb = load_workbook(out)
ws = wb["README_Index"]
wb._sheets.remove(ws)
wb._sheets.insert(0, ws)
wb.save(out)

# Sanity check
xls = pd.ExcelFile(out)
idx = pd.read_excel(out, sheet_name="README_Index", engine="openpyxl")
print(f"[OK] Wrote {out}")
print(f"[OK] Total sheets: {len(xls.sheet_names)}")
print("[OK] Counts by benchmark:")
print(idx.groupby("benchmark")["combined_sheet"].count().to_string())
print("[OK] First sheets:", xls.sheet_names[:12])
