"""Shared utility helpers."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path = "configs/config.yaml") -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        logging.warning(f"Config not found: {p}. Using defaults.")
        return {}
    with open(p) as f:
        return yaml.safe_load(f) or {}


def setup_logging(cfg: dict) -> None:
    level = cfg.get("system", {}).get("log_level", "INFO")
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def ensure_api_key(cfg: dict | None = None) -> None:
    """
    Check the required API key is present for the configured backend.
    - backend=gemini   → needs GEMINI_API_KEY
    - backend=lmstudio → no key needed (local model)

    On Windows PowerShell set the key with:
        $env:GEMINI_API_KEY = "your_key_here"

    On Linux/Mac:
        export GEMINI_API_KEY=your_key_here
    """
    backend = (cfg or {}).get("backend", "gemini")

    if backend == "lmstudio":
        return   # no API key needed for local LM Studio

    # Gemini (default)
    if not os.environ.get("GEMINI_API_KEY"):
        raise EnvironmentError(
            "GEMINI_API_KEY environment variable is not set.\n\n"
            "Windows PowerShell:\n"
            '    $env:GEMINI_API_KEY = "your_key_here"\n\n'
            "Linux / Mac:\n"
            "    export GEMINI_API_KEY=your_key_here\n\n"
            "Get a free key at: https://aistudio.google.com/app/apikey\n\n"
            "Or switch to LM Studio backend by editing configs/config.yaml:\n"
            '    backend: "lmstudio"'
        )


def flatten_report(report) -> dict[str, Any]:
    d: dict[str, Any] = {
        "system":      report.system_name,
        "n":           report.n_samples,
        "accuracy":    report.accuracy,
        "acc_ci_lo":   report.accuracy_ci[0],
        "acc_ci_hi":   report.accuracy_ci[1],
        "macro_f1":    report.macro_f1,
        "f1_ci_lo":    report.macro_f1_ci[0],
        "f1_ci_hi":    report.macro_f1_ci[1],
        "n_uncertain": report.n_uncertain_escalated,
    }
    for m in report.per_class:
        d[f"{m.label}_P"]  = m.precision
        d[f"{m.label}_R"]  = m.recall
        d[f"{m.label}_F1"] = m.f1
    return d
