"""Dataset loading utilities for ADSS.

Supports:
- LegalBench hearsay
- LegalBench contract_nli_confidentiality_of_agreement

The loader normalizes all tasks into the common HearsayExample model used by the
rest of the prototype. Despite the model name, the object is now used as a generic
binary labelled example container.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.data.models import HearsayExample

logger = logging.getLogger(__name__)

YES_VALUES = {"yes", "true", "entailment", "entailed", "entails", "support", "supported", "1"}
NO_VALUES = {"no", "false", "contradiction", "contradicted", "not_entailment", "not entailment", "neutral", "0"}


def _label_to_yes_no(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, (int, float)) and value in (0, 1):
        return "Yes" if int(value) == 1 else "No"
    s = str(value).strip()
    low = s.lower()
    if low in YES_VALUES:
        return "Yes"
    if low in NO_VALUES:
        return "No"
    if low.startswith("yes"):
        return "Yes"
    if low.startswith("no"):
        return "No"
    return None


def _first(row: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    for k in keys:
        if k in row and row[k] not in (None, ""):
            return row[k]
    return default


def _normalise_hearsay(row: dict[str, Any], idx: int, split: str) -> HearsayExample:
    text = _first(row, ["text", "question", "prompt", "input", "narrative", "case", "case_text"], "")
    label = _label_to_yes_no(_first(row, ["label", "answer", "target", "output", "gold", "gold_label"]))
    case_id = str(_first(row, ["case_id", "id", "idx"], f"{split}_{idx:04d}"))
    decision_question = _first(row, ["decision_question", "decision_problem"], None)
    return HearsayExample(case_id=case_id, text=str(text), decision_question=decision_question, label=label, split=split)


def _normalise_contract_nli(row: dict[str, Any], idx: int, split: str) -> HearsayExample:
    # LegalBench contract NLI variants have appeared with slightly different field names.
    contract = _first(row, [
        "premise", "context", "contract", "contract_text", "document", "text", "input", "passage"
    ], "")
    statement = _first(row, [
        "hypothesis", "statement", "claim", "target_statement", "query", "question"
    ], "")

    if statement:
        text = (
            "CONTRACT CONTEXT:\n"
            f"{contract}\n\n"
            "TARGET STATEMENT:\n"
            f"{statement}\n\n"
            "DECISION TASK:\nDetermine whether the contract context entails the target statement."
        )
    else:
        text = str(contract)

    label = _label_to_yes_no(_first(row, ["label", "answer", "target", "output", "gold", "gold_label"]))
    case_id = str(_first(row, ["case_id", "id", "idx"], f"{split}_{idx:04d}"))
    return HearsayExample(case_id=case_id, text=text, decision_question=None, label=label, split=split)


def _normalise_row(row: dict[str, Any], idx: int, split: str, hf_config: str) -> HearsayExample:
    if "contract_nli" in hf_config:
        return _normalise_contract_nli(row, idx, split)
    return _normalise_hearsay(row, idx, split)


def _load_cache(path: Path, split: str, hf_config: str) -> list[HearsayExample]:
    data = json.loads(path.read_text(encoding="utf-8"))
    examples: list[HearsayExample] = []
    for i, row in enumerate(data):
        if row.get("split", split) != split:
            continue
        examples.append(_normalise_row(row, i, split, hf_config))
    logger.info("Loaded %d from cache (%s).", len(examples), split)
    return examples


def _save_cache(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Cache saved → %s", path)


def _load_hf(dataset_cfg: dict, split: str) -> list[dict[str, Any]]:
    from datasets import load_dataset

    repo = dataset_cfg.get("hf_repo", "nguha/legalbench")
    hf_config = dataset_cfg.get("hf_config", "hearsay")
    split_name = dataset_cfg.get(f"{split}_split", split)
    logger.info("Loading %s/%s split=%s from HuggingFace...", repo, hf_config, split_name)

    # Modern datasets no longer supports trust_remote_code for this dataset; avoid it.
    try:
        ds = load_dataset(repo, hf_config, split=split_name)
    except TypeError:
        ds = load_dataset(repo, hf_config, split=split_name)

    rows = [dict(r) for r in ds]
    logger.info("  Loaded %d examples.", len(rows))
    return rows


def load_examples(dataset_cfg: dict, split: str = "test") -> list[HearsayExample]:
    """Load and normalize examples from cache or HuggingFace."""
    hf_config = dataset_cfg.get("hf_config", "hearsay")
    cache_path = Path(dataset_cfg.get("local_cache", f"data/{hf_config}_{split}_cache.json"))

    if cache_path.exists():
        examples = _load_cache(cache_path, split, hf_config)
    else:
        rows = _load_hf(dataset_cfg, split)
        normalised_rows: list[dict[str, Any]] = []
        examples = []
        for i, row in enumerate(rows):
            ex = _normalise_row(row, i, split, hf_config)
            examples.append(ex)
            normalised_rows.append({
                "case_id": ex.case_id,
                "text": ex.text,
                "decision_question": getattr(ex, "decision_question", None),
                "label": ex.label,
                "split": split,
                "source_task": hf_config,
            })
        _save_cache(cache_path, normalised_rows)

    max_samples = dataset_cfg.get("max_samples")
    if max_samples:
        examples = examples[: int(max_samples)]
    return examples
