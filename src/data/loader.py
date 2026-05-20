"""
Dataset loader for LegalBench hearsay task.
Tries local JSON cache first; falls back to HuggingFace Hub.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from src.data.models import HearsayExample

logger = logging.getLogger(__name__)


def _make_id(split: str, idx: int) -> str:
    return f"{split}_{idx:04d}"


def load_from_huggingface(
    repo: str = "nguha/legalbench",
    config: str = "hearsay",
    split: str = "train",
    max_samples: Optional[int] = None,
) -> list[HearsayExample]:
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        raise ImportError("pip install datasets")

    logger.info(f"Loading {repo}/{config} split={split} from HuggingFace...")
    ds = load_dataset(repo, config, split=split, trust_remote_code=True)
    examples: list[HearsayExample] = []
    for idx, row in enumerate(ds):
        if max_samples is not None and idx >= max_samples:
            break
        text  = row.get("text", "") or row.get("input", "")
        label = row.get("answer", None) or row.get("label", None)
        examples.append(HearsayExample(
            case_id=_make_id(split, idx),
            text=text.strip(),
            label=label,
            split=split,
        ))
    logger.info(f"  Loaded {len(examples)} examples.")
    return examples


def load_from_cache(cache_path: str | Path, split: str) -> list[HearsayExample]:
    path = Path(cache_path)
    if not path.exists():
        raise FileNotFoundError(f"Cache not found: {path}")
    with open(path) as f:
        raw = json.load(f)
    return [HearsayExample(**d) for d in raw if d.get("split") == split]


def save_cache(examples: list[HearsayExample], cache_path: str | Path) -> None:
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([e.model_dump() for e in examples], f, indent=2, ensure_ascii=False)
    logger.info(f"Cache saved → {path}")


def load_examples(cfg: dict, split: str = "train") -> list[HearsayExample]:
    """Unified loader: cache → HuggingFace."""
    cache  = cfg.get("local_cache", "data/hearsay_cache.json")
    max_s  = cfg.get("max_samples", None)
    try:
        examples = load_from_cache(cache, split)
        if max_s:
            examples = examples[:max_s]
        logger.info(f"Loaded {len(examples)} from cache ({split}).")
        return examples
    except FileNotFoundError:
        pass

    examples = load_from_huggingface(
        repo=cfg.get("hf_repo", "nguha/legalbench"),
        config=cfg.get("hf_config", "hearsay"),
        split=split,
        max_samples=max_s,
    )
    # merge-write cache
    all_ex: list[HearsayExample] = []
    for s in ("train", "test"):
        try:
            all_ex += load_from_cache(cache, s)
        except FileNotFoundError:
            pass
    all_ids = {e.case_id for e in all_ex}
    all_ex += [e for e in examples if e.case_id not in all_ids]
    save_cache(all_ex, cache)
    return examples
