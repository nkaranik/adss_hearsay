
"""
Automated HITL contestability simulations for ADSS.

Regimes:
- Oracle-guided: uses gold label to estimate upper-bound correctability.
- Confidence-guided: edits low-confidence arguments/relations without gold.

Metrics:
- DFR: Decision Flip Rate
- CFR: Corrective Flip Rate
- MEC: Minimal Edit Count
- SSM: Score Shift Magnitude
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import csv
import json
import random

from src.data.models import Decision, RelationType
from src.qbaf.solver import make_decision


def _dec_value(d: Any) -> str:
    return d.value if hasattr(d, "value") else str(d)


def _label_from_sigma(sigma: float, cfg: dict) -> str:
    return make_decision(sigma, cfg)[0]


def _binary_label(label: str) -> str:
    return "No" if label == "UNCERTAIN" else label


def _is_correct(pred: Any) -> bool:
    gold = getattr(pred, "gold_label", None)
    if gold not in ("Yes", "No"):
        return False
    return _binary_label(_dec_value(pred.decision)) == gold


@dataclass
class SimulationDetail:
    case_id: str
    regime: str
    original_decision: str
    new_decision: str
    gold_label: str | None
    original_sigma: float
    new_sigma: float
    flipped: bool
    corrected: bool
    edits_used: int
    score_shift: float


@dataclass
class ContestabilityMetrics:
    regime: str
    n_total: int
    decision_flip_rate: float
    corrective_flip_rate: float
    minimal_edit_count: float
    score_shift_magnitude: float
    details: list[SimulationDetail]


def _solve(graph: Any, solver: Any, cfg: dict, case_id: str) -> tuple[float, str]:
    out = solver.solve(graph, case_id)
    sigma = out.sigma_phi
    return sigma, _label_from_sigma(sigma, cfg)


def _candidate_arg_ids(graph: Any) -> list[str]:
    return [nid for nid, node in graph.nodes.items() if not getattr(node, "is_claim", False) and nid != "phi"]


def _edge_to_phi_type(graph: Any, arg_id: str):
    for e in graph.edges:
        if e.source == arg_id and e.target == "phi":
            return e.type
    return None


def _apply_oracle_edit(graph: Any, gold: str | None, edit_index: int) -> bool:
    """Apply one greedy oracle edit. Returns True if an edit was applied."""
    if gold not in ("Yes", "No"):
        return False

    want_support = gold == "Yes"
    args = _candidate_arg_ids(graph)
    if not args:
        return False

    # Pick the strongest counter-direction argument first.
    scored = []
    for aid in args:
        node = graph.nodes[aid]
        etype = _edge_to_phi_type(graph, aid)
        tau = float(getattr(node, "tau", 0.5))
        scored.append((tau, aid, etype))
    scored.sort(reverse=True)

    for _, aid, etype in scored:
        if want_support and etype == RelationType.ATTACK:
            for e in graph.edges:
                if e.source == aid and e.target == "phi":
                    e.type = RelationType.SUPPORT
                    return True
        if (not want_support) and etype == RelationType.SUPPORT:
            for e in graph.edges:
                if e.source == aid and e.target == "phi":
                    e.type = RelationType.ATTACK
                    return True

    # If no relation flip is available, adjust tau in the desired direction.
    aid = scored[edit_index % len(scored)][1]
    node = graph.nodes[aid]
    node.tau = 1.0 if want_support else 0.1
    node.sigma = node.tau
    return True


def _apply_confidence_edit(graph: Any, rng: random.Random) -> bool:
    """Blindly target lowest-confidence edge/node."""
    candidates = []
    for e in graph.edges:
        conf = float(getattr(e, "confidence", 1.0))
        candidates.append((conf, "edge", e))
    for aid in _candidate_arg_ids(graph):
        node = graph.nodes[aid]
        tau = float(getattr(node, "tau", 0.5))
        candidates.append((abs(tau - 0.5), "node", node))
    if not candidates:
        return False
    candidates.sort(key=lambda x: x[0])
    _, kind, obj = candidates[0]
    if kind == "edge":
        obj.type = RelationType.ATTACK if obj.type == RelationType.SUPPORT else RelationType.SUPPORT
    else:
        # Move low-confidence argument away from indecision.
        obj.tau = 0.1 if rng.random() < 0.5 else 1.0
        obj.sigma = obj.tau
    return True


def simulate_one(pred: Any, solver: Any, cfg: dict, regime: str, max_edits: int = 3, seed: int = 42) -> SimulationDetail:
    rng = random.Random(seed)
    graph = pred.solver_output.graph.model_copy(deep=True)
    original_sigma = float(pred.sigma_phi)
    original_decision = _dec_value(pred.decision)
    original_binary = _binary_label(original_decision)
    gold = getattr(pred, "gold_label", None)

    new_sigma = original_sigma
    new_decision = original_decision
    edits_used = 0

    for i in range(max_edits):
        if regime == "oracle":
            applied = _apply_oracle_edit(graph, gold, i)
        elif regime == "confidence":
            applied = _apply_confidence_edit(graph, rng)
        else:
            raise ValueError(f"Unknown regime: {regime}")
        if not applied:
            break
        edits_used += 1
        new_sigma, new_decision = _solve(graph, solver, cfg, pred.case_id)
        if _binary_label(new_decision) != original_binary:
            break

    new_binary = _binary_label(new_decision)
    flipped = new_binary != original_binary
    corrected = bool(gold in ("Yes", "No") and original_binary != gold and new_binary == gold)
    return SimulationDetail(
        case_id=pred.case_id,
        regime=regime,
        original_decision=original_decision,
        new_decision=new_decision,
        gold_label=gold,
        original_sigma=original_sigma,
        new_sigma=new_sigma,
        flipped=flipped,
        corrected=corrected,
        edits_used=edits_used,
        score_shift=abs(new_sigma - original_sigma),
    )


def run_contestability_simulation(
    predictions: list[Any],
    solver: Any,
    cfg: dict,
    regime: str,
    max_edits: int = 3,
    seed: int = 42,
) -> ContestabilityMetrics:
    details = []
    for i, pred in enumerate(predictions):
        if getattr(pred, "solver_output", None) is None:
            continue
        details.append(simulate_one(pred, solver, cfg, regime, max_edits, seed + i))

    n = len(details)
    if n == 0:
        return ContestabilityMetrics(regime, 0, 0.0, 0.0, 0.0, 0.0, [])

    flipped = [d for d in details if d.flipped]
    corrected = [d for d in details if d.corrected]
    return ContestabilityMetrics(
        regime=regime,
        n_total=n,
        decision_flip_rate=len(flipped) / n,
        corrective_flip_rate=len(corrected) / n,
        minimal_edit_count=sum(d.edits_used for d in flipped) / len(flipped) if flipped else 0.0,
        score_shift_magnitude=sum(d.score_shift for d in details) / n,
        details=details,
    )


def write_contestability_report(results: list[ContestabilityMetrics], out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "contestability_metrics.json").write_text(
        json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8"
    )
    with (out / "contestability_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "regime", "n_total", "decision_flip_rate", "corrective_flip_rate",
            "minimal_edit_count", "score_shift_magnitude",
        ])
        w.writeheader()
        for r in results:
            w.writerow({
                "regime": r.regime,
                "n_total": r.n_total,
                "decision_flip_rate": r.decision_flip_rate,
                "corrective_flip_rate": r.corrective_flip_rate,
                "minimal_edit_count": r.minimal_edit_count,
                "score_shift_magnitude": r.score_shift_magnitude,
            })
