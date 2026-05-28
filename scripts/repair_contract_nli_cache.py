#!/usr/bin/env python3
"""Repair LegalBench Contract NLI confidentiality cache for ADSS.

Problem fixed:
The cached examples may contain only the contract context in `text` and have
`decision_question=None`. For Contract NLI, the model must see BOTH:
  1) CONTRACT CONTEXT
  2) EXACT TARGET STATEMENT / hypothesis
otherwise it tends to answer Yes whenever confidentiality language appears.

Usage from project root:
python scripts/repair_contract_nli_cache.py \
  --cache data/contract_nli_confidentiality_cache.json \
  --backup --force
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

DEFAULT_TARGET = (
    "The contract entails that the receiving party must keep confidential "
    "the existence and/or terms of the agreement itself."
)

STRICT_DECISION_QUESTION = (
    "Does the CONTRACT CONTEXT logically entail the EXACT TARGET STATEMENT, "
    "with every material element matched and no missing limitation, exception, "
    "party-scope mismatch, information-type mismatch, purpose limitation, "
    "recipient-scope mismatch, or disclosure carve-out?"
)

INSTRUCTIONS = (
    "Important Contract NLI rules:\n"
    "- Do NOT answer Yes merely because the contract contains confidentiality language.\n"
    "- Support requires exact entailment of the TARGET STATEMENT.\n"
    "- If the clause is only related to confidentiality, but differs in party scope, "
    "information type, obligation type, recipient scope, purpose limitation, or exceptions, "
    "that mismatch ATTACKS entailment.\n"
    "- Carve-outs and permitted disclosures are attacks when the target statement is broader "
    "or absolute.\n"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="data/contract_nli_confidentiality_cache.json")
    p.add_argument("--target-statement", default=DEFAULT_TARGET)
    p.add_argument("--backup", action="store_true")
    p.add_argument("--force", action="store_true", help="Rewrite even if examples already look repaired.")
    return p.parse_args()


def is_repaired(text: str) -> bool:
    return "CONTRACT CONTEXT:" in text and "TARGET STATEMENT:" in text


def main() -> None:
    args = parse_args()
    path = Path(args.cache)
    if not path.exists():
        raise FileNotFoundError(path)

    if args.backup:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = path.with_suffix(path.suffix + f".bak_{stamp}")
        shutil.copy2(path, backup)
        print(f"[OK] Backup written: {backup}")

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Expected cache JSON to be a list of examples")

    changed = 0
    skipped = 0
    for ex in data:
        source_task = str(ex.get("source_task", ""))
        if "contract_nli_confidentiality" not in source_task:
            skipped += 1
            continue

        current_text = str(ex.get("text", ""))
        if is_repaired(current_text) and not args.force:
            skipped += 1
            continue

        # Preserve original context once. If force is used after a previous repair,
        # prefer raw_contract_context to avoid nesting repaired text repeatedly.
        original_context = ex.get("raw_contract_context") or current_text
        ex["raw_contract_context"] = original_context
        ex["target_statement"] = args.target_statement
        ex["decision_question"] = STRICT_DECISION_QUESTION
        ex["text"] = (
            "CONTRACT CONTEXT:\n"
            f"{original_context.strip()}\n\n"
            "TARGET STATEMENT:\n"
            f"{args.target_statement.strip()}\n\n"
            f"{INSTRUCTIONS}"
        )
        ex["input_format_version"] = "contract_nli_exact_target_v2"
        changed += 1

    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Repaired {changed} examples in {path}")
    print(f"[INFO] Skipped {skipped} examples")
    print("[INFO] Target statement:", args.target_statement)
    print("[INFO] Decision question:", STRICT_DECISION_QUESTION)


if __name__ == "__main__":
    main()
