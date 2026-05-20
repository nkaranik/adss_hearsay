"""
Contestability evaluation module.
Computes DFR, CFR, MEC, SSM under simulated interventions.

Two simulation regimes:
  oracle-guided    — interventions selected using gold labels (upper bound)
  confidence-guided — interventions selected from low-confidence args/edges (blind)
"""
from __future__ import annotations

import logging
import random
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional

from src.data.models import ADSSPrediction, Decision, HITLIntervention, RelationType

logger = logging.getLogger(__name__)


@dataclass
class ContestabilityResult:
    regime:   str          # "oracle" | "confidence"
    dfr:      float = 0.0  # Decision Flip Rate
    cfr:      float = 0.0  # Corrective Flip Rate
    mec:      float = 0.0  # Mean Minimal Edit Count
    ssm:      float = 0.0  # Mean Score Shift Magnitude
    n_total:  int   = 0
    n_flipped: int  = 0
    n_corrected: int = 0
    details:  list  = field(default_factory=list)


def _apply_tau_edit(graph, arg_id: str, new_tau: float):
    """Mutate tau in-place on a deep-copied graph."""
    if arg_id in graph.nodes:
        graph.nodes[arg_id].tau   = max(0.1, min(1.0, new_tau))
        graph.nodes[arg_id].sigma = graph.nodes[arg_id].tau


def _resolve_decision(sigma_phi: float, cfg: dict) -> Decision:
    from src.qbaf.solver import make_decision
    return Decision(make_decision(sigma_phi, cfg)[0])


def _re_solve(graph, solver, case_id: str) -> float:
    out = solver.solve(graph, case_id)
    return out.sigma_phi


# ── Oracle-guided simulation ──────────────────────────────────────────────────

def _oracle_intervention(
    pred: ADSSPrediction,
    solver,
    cfg: dict,
    max_edits: int = 3,
) -> tuple[float, int, float]:
    """
    Try to flip the decision toward the gold label using oracle knowledge.
    Returns (new_sigma_phi, edits_used, score_shift).
    If already correct, returns original sigma unchanged.
    """
    if pred.gold_label is None or pred.solver_output is None:
        return pred.sigma_phi, 0, 0.0

    gold     = pred.gold_label          # "Yes" or "No"
    is_wrong = not pred.is_correct()
    original_sigma = pred.sigma_phi

    graph = pred.solver_output.graph.model_copy(deep=True)
    non_phi = [nid for nid in graph.nodes if nid != "phi"]

    edits = 0
    for edit_num in range(1, max_edits + 1):
        if not non_phi:
            break

        # Find the arg that most needs changing
        # For gold=Yes: boost support args, reduce attack args
        # For gold=No:  boost attack args, reduce support args
        arg_ids_support = graph.supporters_of("phi")
        arg_ids_attack  = graph.attackers_of("phi")

        if gold == "Yes":
            # boost best support, reduce best attack
            candidates = (
                [(aid, graph.nodes[aid].tau, "boost") for aid in arg_ids_support]
                + [(aid, graph.nodes[aid].tau, "reduce") for aid in arg_ids_attack]
            )
        else:
            candidates = (
                [(aid, graph.nodes[aid].tau, "reduce") for aid in arg_ids_support]
                + [(aid, graph.nodes[aid].tau, "boost") for aid in arg_ids_attack]
            )

        if not candidates:
            break

        # Sort: boost targets with lowest tau first, reduce with highest tau first
        def sort_key(c):
            _, tau, action = c
            return tau if action == "boost" else -tau

        candidates.sort(key=sort_key)
        target_id, old_tau, action = candidates[0]
        new_tau = 0.95 if action == "boost" else 0.1
        _apply_tau_edit(graph, target_id, new_tau)
        edits += 1

        new_sigma = _re_solve(graph, solver, pred.case_id)
        new_dec   = _resolve_decision(new_sigma, cfg)

        if new_dec.value == gold:
            shift = abs(new_sigma - original_sigma)
            return new_sigma, edits, shift

    # After max_edits, return best achieved
    final_sigma = _re_solve(graph, solver, pred.case_id)
    return final_sigma, edits, abs(final_sigma - original_sigma)


# ── Confidence-guided simulation ──────────────────────────────────────────────

def _confidence_intervention(
    pred: ADSSPrediction,
    solver,
    cfg: dict,
    confidence_threshold: float = 0.5,
    max_edits: int = 3,
) -> tuple[float, int, float]:
    """
    Simulate edits based on low-confidence arguments (no gold label access).
    Reduce τ of low-confidence args, remove low-confidence edges.
    """
    if pred.solver_output is None:
        return pred.sigma_phi, 0, 0.0

    original_sigma = pred.sigma_phi
    graph = pred.solver_output.graph.model_copy(deep=True)

    # Low-confidence args (from extraction confidence)
    low_conf_args = [
        a for a in pred.extraction.arguments
        if a.confidence < confidence_threshold
    ]
    # Low-confidence edges
    low_conf_edges = [
        e for e in graph.edges
        if e.confidence < confidence_threshold and e.target == "phi"
    ]

    edits = 0
    # Reduce tau of low-confidence args
    for arg in low_conf_args[:max_edits]:
        if arg.id in graph.nodes:
            old = graph.nodes[arg.id].tau
            _apply_tau_edit(graph, arg.id, old * 0.5)
            edits += 1
            if edits >= max_edits:
                break

    # Remove low-confidence edges if budget remains
    for edge in low_conf_edges:
        if edits >= max_edits:
            break
        graph.edges = [e for e in graph.edges
                       if not (e.source == edge.source and e.target == edge.target)]
        edits += 1

    final_sigma = _re_solve(graph, solver, pred.case_id)
    shift = abs(final_sigma - original_sigma)
    return final_sigma, edits, shift


# ── Main evaluation ───────────────────────────────────────────────────────────

def compute_contestability(
    predictions: list[ADSSPrediction],
    solver,
    cfg: dict,
    regime: str = "oracle",
    max_edits: int = 3,
    seed: int = 42,
) -> ContestabilityResult:
    """
    Compute DFR, CFR, MEC, SSM for a list of predictions.

    regime: "oracle" | "confidence"
    """
    random.seed(seed)
    result = ContestabilityResult(regime=regime)

    labelled = [p for p in predictions
                if p.gold_label is not None and p.solver_output is not None]
    result.n_total = len(labelled)
    if not labelled:
        logger.warning("No labelled predictions with solver output for contestability eval.")
        return result

    total_edits = 0
    total_shift = 0.0
    n_flipped   = 0
    n_corrected = 0

    for pred in labelled:
        orig_dec   = pred.decision
        orig_sigma = pred.sigma_phi
        orig_correct = pred.is_correct()

        if regime == "oracle":
            new_sigma, edits, shift = _oracle_intervention(pred, solver, cfg, max_edits)
        else:
            new_sigma, edits, shift = _confidence_intervention(
                pred, solver, cfg, max_edits=max_edits
            )

        new_dec     = _resolve_decision(new_sigma, cfg)
        flipped     = new_dec != orig_dec and new_dec != Decision.UNCERTAIN
        corrected   = (not orig_correct) and (new_dec.value == pred.gold_label)

        total_edits  += edits
        total_shift  += shift
        n_flipped    += int(flipped)
        n_corrected  += int(corrected)

        result.details.append({
            "case_id":      pred.case_id,
            "gold":         pred.gold_label,
            "orig_dec":     orig_dec.value,
            "orig_sigma":   orig_sigma,
            "new_dec":      new_dec.value,
            "new_sigma":    new_sigma,
            "edits":        edits,
            "shift":        shift,
            "flipped":      flipped,
            "corrected":    corrected,
        })

    n_wrong = sum(1 for p in labelled if not p.is_correct())

    result.dfr = n_flipped / result.n_total if result.n_total else 0.0
    result.cfr = n_corrected / n_wrong if n_wrong else 0.0
    result.mec = total_edits / result.n_total if result.n_total else 0.0
    result.ssm = total_shift / result.n_total if result.n_total else 0.0
    result.n_flipped   = n_flipped
    result.n_corrected = n_corrected

    logger.info(
        f"[{regime}] DFR={result.dfr:.3f} CFR={result.cfr:.3f} "
        f"MEC={result.mec:.3f} SSM={result.ssm:.3f}"
    )
    return result


def print_contestability(r: ContestabilityResult) -> None:
    print(f"\n  Contestability [{r.regime}]")
    print(f"    n={r.n_total}  flipped={r.n_flipped}  corrected={r.n_corrected}")
    print(f"    DFR={r.dfr:.3f}  CFR={r.cfr:.3f}  MEC={r.mec:.3f}  SSM={r.ssm:.3f}")
