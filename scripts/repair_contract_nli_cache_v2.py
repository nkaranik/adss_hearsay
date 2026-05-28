#!/usr/bin/env python3
"""Repair Contract NLI cache, v2: keep raw contract text in `text` and store target separately.

Why v2?
The first repair placed CONTRACT CONTEXT / TARGET STATEMENT directly inside `text`.
The evaluation loader/pipeline already formats Contract NLI examples when `target_statement`
is present, so putting the formatted block inside `text` caused duplicated prompts:
CONTRACT CONTEXT: CONTRACT CONTEXT: ... TARGET STATEMENT: ... TARGET STATEMENT: ...

This v2 repair keeps:
- text = raw contract context only
- target_statement = benchmark-level exact target
- decision_question = strict entailment decision problem

Usage:
python scripts/repair_contract_nli_cache_v2.py --cache data/contract_nli_confidentiality_cache.json --backup --force
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

DEFAULT_TARGET = (
    "The contract entails that the relevant party must keep confidential the existence, "
    "terms, subject matter, or status of the agreement, negotiations, discussions, or "
    "transaction between the parties."
)

STRICT_DECISION_QUESTION = (
    "Does the CONTRACT CONTEXT logically entail the EXACT TARGET STATEMENT, "
    "with every material element matched and no missing limitation, exception, "
    "party-scope mismatch, information-type mismatch, purpose limitation, "
    "recipient-scope mismatch, or disclosure carve-out?"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="data/contract_nli_confidentiality_cache.json")
    p.add_argument("--target-statement", default=DEFAULT_TARGET)
    p.add_argument("--backup", action="store_true")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def strip_repaired_blocks(text: str) -> str:
    """Best-effort fallback if raw_contract_context is unavailable."""
    s = text or ""
    if "CONTRACT CONTEXT:" in s:
        s = s.split("CONTRACT CONTEXT:", 1)[1]
    if "TARGET STATEMENT:" in s:
        s = s.split("TARGET STATEMENT:", 1)[0]
    return s.strip()


def main() -> None:
    args = parse_args()
    path = Path(args.cache)
    if not path.exists():
        raise FileNotFoundError(path)

    if args.backup:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = path.with_suffix(path.suffix + f".bak_v2_{stamp}")
        shutil.copy2(path, backup)
        print(f"[OK] Backup written: {backup}")

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Expected cache JSON to be a list")

    changed = 0
    for ex in data:
        if "contract_nli_confidentiality" not in str(ex.get("source_task", "")):
            continue
        raw = ex.get("raw_contract_context")
        if not raw:
            raw = strip_repaired_blocks(str(ex.get("text", "")))
        ex["raw_contract_context"] = raw
        ex["text"] = raw
        ex["target_statement"] = args.target_statement
        ex["decision_question"] = STRICT_DECISION_QUESTION
        ex["input_format_version"] = "contract_nli_target_separate_v2"
        changed += 1

    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Repaired {changed} Contract NLI examples in {path}")
    print("[INFO] text now contains raw contract context only.")
    print("[INFO] target_statement:", args.target_statement)
    print("[INFO] decision_question:", STRICT_DECISION_QUESTION)


if __name__ == "__main__":
    main()
