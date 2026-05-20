"""
Error analysis: categorise misclassified predictions into the 4 paper categories.
  1. Argument omission
  2. Relation error
  3. Strength miscalibration
  4. Threshold / uncertainty failure
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from src.data.models import ADSSPrediction, Decision

logger = logging.getLogger(__name__)

CATEGORIES = [
    "argument_omission",
    "relation_error",
    "strength_miscalibration",
    "threshold_uncertainty_failure",
]


@dataclass
class ErrorRecord:
    case_id:   str
    gold:      str
    predicted: str
    sigma_phi: float
    category:  str
    reason:    str


def _classify_error(pred: ADSSPrediction) -> tuple[str, str]:
    """
    Return (category, reason) for a misclassified prediction.
    Categories are mutually exclusive; first matching rule wins.
    """
    n_args = len(pred.extraction.arguments)
    sigma  = pred.sigma_phi
    gold   = pred.gold_label or "?"

    # 1. Argument omission — too few args extracted
    if n_args == 0:
        return "argument_omission", "No arguments extracted at all."
    if n_args <= 1:
        return "argument_omission", f"Only {n_args} argument(s) extracted; insufficient coverage."

    # 2. Threshold / uncertainty failure — sigma very close to boundary
    low, high = 0.40, 0.60   # slightly wider than UAE band for error analysis
    if low <= sigma <= high:
        return "threshold_uncertainty_failure", (
            f"σ(φ)={sigma:.4f} falls in borderline zone [{low},{high}]; "
            "escalation should have applied."
        )

    # 3. Relation error — wrong stances dominate
    if pred.solver_output:
        supporters = pred.solver_output.graph.supporters_of("phi")
        attackers  = pred.solver_output.graph.attackers_of("phi")
        if gold == "Yes" and len(attackers) > len(supporters):
            return "relation_error", (
                f"Gold=Yes but {len(attackers)} attack(s) > {len(supporters)} support(s); "
                "stances likely misclassified."
            )
        if gold == "No" and len(supporters) > len(attackers):
            return "relation_error", (
                f"Gold=No but {len(supporters)} support(s) > {len(attackers)} attack(s); "
                "stances likely misclassified."
            )

    # 4. Strength miscalibration — stances are in the right direction but sigma is wrong
    if pred.strengths:
        tau_map = {s.argument_id: s.tau for s in pred.strengths}
        supporters = pred.solver_output.graph.supporters_of("phi") if pred.solver_output else []
        attackers  = pred.solver_output.graph.attackers_of("phi")  if pred.solver_output else []
        sup_taus   = [tau_map.get(aid, 0.5) for aid in supporters]
        att_taus   = [tau_map.get(aid, 0.5) for aid in attackers]
        mean_sup   = sum(sup_taus) / len(sup_taus) if sup_taus else 0.5
        mean_att   = sum(att_taus) / len(att_taus) if att_taus else 0.5
        if gold == "Yes" and mean_att > mean_sup:
            return "strength_miscalibration", (
                f"Gold=Yes but mean τ(attack)={mean_att:.3f} > mean τ(support)={mean_sup:.3f}; "
                "attackers over-scored."
            )
        if gold == "No" and mean_sup > mean_att:
            return "strength_miscalibration", (
                f"Gold=No but mean τ(support)={mean_sup:.3f} > mean τ(attack)={mean_att:.3f}; "
                "supporters over-scored."
            )

    # Default fallback
    return "argument_omission", (
        f"No specific pattern identified. n_args={n_args}, σ={sigma:.4f}."
    )


def analyse_errors(
    predictions: list[ADSSPrediction],
    output_dir: str | Path | None = None,
) -> dict[str, float]:
    """
    Classify each misclassified prediction and return proportions per category.
    """
    errors: list[ErrorRecord] = []

    for pred in predictions:
        if pred.gold_label is None:
            continue
        if pred.is_correct():
            continue   # only errors

        cat, reason = _classify_error(pred)
        dec = pred.decision.value if pred.decision != Decision.UNCERTAIN else "No"
        errors.append(ErrorRecord(
            case_id=pred.case_id,
            gold=pred.gold_label,
            predicted=dec,
            sigma_phi=pred.sigma_phi,
            category=cat,
            reason=reason,
        ))

    if not errors:
        logger.info("No errors to analyse.")
        return {c: 0.0 for c in CATEGORIES}

    total = len(errors)
    counts = {c: sum(1 for e in errors if e.category == c) for c in CATEGORIES}
    proportions = {c: counts[c] / total for c in CATEGORIES}

    logger.info(f"Error analysis: {total} errors total")
    for c in CATEGORIES:
        logger.info(f"  {c}: {counts[c]} ({proportions[c]:.1%})")

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "error_analysis.json", "w", encoding="utf-8") as f:
            json.dump({
                "total_errors": total,
                "counts": counts,
                "proportions": {k: round(v, 4) for k, v in proportions.items()},
                "records": [
                    {"case_id": e.case_id, "gold": e.gold,
                     "predicted": e.predicted, "sigma_phi": e.sigma_phi,
                     "category": e.category, "reason": e.reason}
                    for e in errors
                ],
            }, f, indent=2)

    return proportions


def print_error_analysis(proportions: dict[str, float]) -> None:
    labels = {
        "argument_omission":            "Argument omission",
        "relation_error":               "Relation error",
        "strength_miscalibration":      "Strength miscalibration",
        "threshold_uncertainty_failure":"Threshold / uncertainty failure",
    }
    print("\n  Error Analysis")
    for cat, prop in proportions.items():
        print(f"    {labels.get(cat, cat):40s}  {prop:.1%}")
