
"""Structured error analysis for ADSS misclassifications."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import csv
import json

CATEGORIES = (
    "argument_omission",
    "relation_error",
    "strength_miscalibration",
    "threshold_uncertainty_failure",
)


@dataclass
class ErrorCase:
    case_id: str
    gold_label: str
    predicted_label: str
    sigma_phi: float
    category: str
    reason: str


def _dval(d: Any) -> str:
    return d.value if hasattr(d, "value") else str(d)


def _binary(label: str) -> str:
    return "No" if label == "UNCERTAIN" else label


def categorize_error(pred: Any, band: tuple[float, float] = (0.45, 0.55)) -> ErrorCase | None:
    gold = getattr(pred, "gold_label", None)
    if gold not in ("Yes", "No"):
        return None
    predicted = _dval(pred.decision)
    if _binary(predicted) == gold:
        return None

    extraction = getattr(pred, "extraction", None)
    args = getattr(extraction, "arguments", []) if extraction else []
    rels = getattr(extraction, "relations", []) if extraction else []
    strengths = getattr(pred, "strengths", []) or []
    sigma = float(getattr(pred, "sigma_phi", 0.5))

    if band[0] <= sigma <= band[1] and predicted != "UNCERTAIN":
        category = "threshold_uncertainty_failure"
        reason = "Case score lies inside uncertainty band but was not flagged as uncertain."
    elif len(args) == 0 or (hasattr(extraction, "n_raw_arguments") and extraction.n_raw_arguments > len(args)):
        category = "argument_omission"
        reason = "No arguments were retained, or relevant arguments were filtered out as neutral/low-confidence."
    elif len(rels) == 0 or not any(getattr(r, "target", None) == "phi" for r in rels):
        category = "relation_error"
        reason = "No usable relations to phi were produced."
    elif strengths:
        avg_tau = sum(float(getattr(s, "tau", 0.5)) for s in strengths) / len(strengths)
        if avg_tau < 0.35 or avg_tau > 0.9:
            category = "strength_miscalibration"
            reason = f"Average tau appears extreme or implausible: {avg_tau:.3f}."
        else:
            category = "relation_error"
            reason = "Arguments exist with plausible strengths; relation polarity/structure is the likely failure."
    else:
        category = "strength_miscalibration"
        reason = "Arguments exist but no strength scores were available."

    return ErrorCase(pred.case_id, gold, predicted, sigma, category, reason)


def analyse_errors(predictions: list[Any], band: tuple[float, float] = (0.45, 0.55)) -> tuple[list[ErrorCase], dict[str, float]]:
    errors = [e for p in predictions if (e := categorize_error(p, band)) is not None]
    if not errors:
        return [], {c: 0.0 for c in CATEGORIES}
    proportions = {c: sum(e.category == c for e in errors) / len(errors) for c in CATEGORIES}
    return errors, proportions


def write_error_report(errors: list[ErrorCase], proportions: dict[str, float], out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "error_analysis.json").write_text(json.dumps({
        "proportions": proportions,
        "errors": [asdict(e) for e in errors],
    }, indent=2), encoding="utf-8")
    with (out / "error_cases.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["case_id", "gold_label", "predicted_label", "sigma_phi", "category", "reason"])
        w.writeheader()
        for e in errors:
            w.writerow(asdict(e))
