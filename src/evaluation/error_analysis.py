"""Structured error analysis for ADSS misclassifications.

Categories aligned with the paper:
1. argument_omission
2. relation_error
3. strength_miscalibration
4. alternative_claim_alignment_error
5. threshold_uncertainty_failure
6. task_transfer_failure
"""
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
    "alternative_claim_alignment_error",
    "threshold_uncertainty_failure",
    "task_transfer_failure",
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


def _safe_lower(x: Any) -> str:
    return str(x or "").lower()


def _claim_text(pred: Any) -> str:
    extraction = getattr(pred, "extraction", None)
    claim = getattr(extraction, "claim", None) if extraction else None
    return str(getattr(claim, "text", "") or "")


def _domain(pred: Any) -> str:
    return _safe_lower(getattr(pred, "task_name", "") or getattr(pred, "domain", ""))


def _looks_like_alignment_error(pred: Any) -> bool:
    extraction = getattr(pred, "extraction", None)
    if extraction is None:
        return False
    claim = _safe_lower(_claim_text(pred))
    args = getattr(extraction, "arguments", []) or []
    if not args:
        return False
    texts = " ".join(_safe_lower(getattr(a, "text", "")) + " " + _safe_lower(getattr(a, "legal_rule", "")) for a in args)
    if "hearsay" in claim or "801" in claim:
        markers = ["hearsay", "out-of-court", "out of court", "truth", "matter asserted", "offered", "notice", "verbal act", "party opponent", "801", "assertion", "statement", "declarant"]
        return not any(m in texts for m in markers)
    if "contract" in claim or "confidential" in claim:
        markers = ["contract", "agreement", "clause", "confidential", "confidentiality", "obligation", "entail", "target", "provision", "party", "disclosure", "recipient"]
        return not any(m in texts for m in markers)
    return False


def _looks_like_task_transfer_failure(pred: Any) -> bool:
    extraction = getattr(pred, "extraction", None)
    if extraction is None:
        return False
    claim = _safe_lower(_claim_text(pred))
    args = getattr(extraction, "arguments", []) or []
    texts = " ".join(_safe_lower(getattr(a, "text", "")) + " " + _safe_lower(getattr(a, "legal_rule", "")) for a in args)
    # Contract NLI case contaminated by hearsay/evidence reasoning.
    if "contract" in claim or "confidential" in claim:
        hearsay_markers = ["hearsay", "out-of-court", "matter asserted", "fre 801", "declarant", "offered for truth"]
        return any(m in texts for m in hearsay_markers)
    return False


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

    if _looks_like_task_transfer_failure(pred):
        category = "task_transfer_failure"
        reason = "Contract/general task appears contaminated by reasoning patterns from another task."
    elif band[0] <= sigma <= band[1] and predicted != "UNCERTAIN":
        category = "threshold_uncertainty_failure"
        reason = "Case score lies inside uncertainty band but was not flagged as uncertain."
    elif len(args) == 0:
        category = "argument_omission"
        reason = "No relevant arguments were retained for the target claim."
    elif _looks_like_alignment_error(pred):
        category = "alternative_claim_alignment_error"
        reason = "Arguments appear aligned with a different issue rather than the formal target claim."
    elif len(rels) == 0 or not any(getattr(r, "target", None) == "phi" for r in rels):
        category = "relation_error"
        reason = "No usable direct relation to phi was produced."
    elif strengths:
        tau_values = [float(getattr(s, "tau", 0.5)) for s in strengths]
        avg_tau = sum(tau_values) / len(tau_values)
        if avg_tau < 0.30 or avg_tau > 0.90:
            category = "strength_miscalibration"
            reason = f"Average tau appears extreme for a misclassified case: {avg_tau:.3f}."
        else:
            category = "relation_error"
            reason = "Arguments exist with non-extreme strengths; relation polarity or graph structure is the likely failure."
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
    (out / "error_analysis.json").write_text(json.dumps({"proportions": proportions, "errors": [asdict(e) for e in errors]}, indent=2), encoding="utf-8")
    with (out / "error_cases.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["case_id", "gold_label", "predicted_label", "sigma_phi", "category", "reason"])
        w.writeheader()
        for e in errors:
            w.writerow(asdict(e))
