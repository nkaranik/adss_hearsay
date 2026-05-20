"""
Robustness evaluation: controlled perturbation of mined argument graphs.
Tests degradation under: arg removal, relation flipping, tau noise, low-conf pruning.
"""
from __future__ import annotations

import logging
import math
import random
from copy import deepcopy
from dataclasses import dataclass, field

from src.data.models import ADSSPrediction, Decision, QBAFEdge, RelationType

logger = logging.getLogger(__name__)


@dataclass
class RobustnessResult:
    perturbation:  str
    p_level:       float
    accuracy:      float
    macro_f1:      float
    mean_score_shift: float   # E[|σ_p - σ_0|]
    n_samples:     int


def _accuracy(golds, preds):
    return sum(g == p for g, p in zip(golds, preds)) / len(golds) if golds else 0.0


def _macro_f1(golds, preds, labels=("Yes", "No")):
    f1s = []
    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(golds, preds))
        fp = sum(g != label and p == label for g, p in zip(golds, preds))
        fn = sum(g == label and p != label for g, p in zip(golds, preds))
        pr = tp / (tp + fp) if (tp + fp) else 0.0
        re = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * pr * re / (pr + re) if (pr + re) else 0.0)
    return sum(f1s) / len(f1s)


def _perturb_and_solve(pred: ADSSPrediction, perturbation: str,
                        p: float, solver, cfg: dict, rng: random.Random) -> float:
    """Apply perturbation to graph and re-solve. Returns new sigma_phi."""
    if pred.solver_output is None:
        return pred.sigma_phi

    graph = pred.solver_output.graph.model_copy(deep=True)
    non_phi = [nid for nid in graph.nodes if nid != "phi"]

    if perturbation == "arg_removal" and non_phi:
        n_remove = max(0, round(p * len(non_phi)))
        to_remove = set(rng.sample(non_phi, min(n_remove, len(non_phi))))
        graph.nodes  = {k: v for k, v in graph.nodes.items() if k not in to_remove}
        graph.edges  = [e for e in graph.edges
                        if e.source not in to_remove and e.target not in to_remove]

    elif perturbation == "relation_flip" and graph.edges:
        n_flip = max(0, round(p * len(graph.edges)))
        flip_idx = set(rng.sample(range(len(graph.edges)), min(n_flip, len(graph.edges))))
        for i in flip_idx:
            e = graph.edges[i]
            if e.type == RelationType.SUPPORT:
                graph.edges[i] = e.model_copy(update={"type": RelationType.ATTACK})
            elif e.type == RelationType.ATTACK:
                graph.edges[i] = e.model_copy(update={"type": RelationType.SUPPORT})

    elif perturbation == "tau_noise":
        for nid in non_phi:
            if nid in graph.nodes:
                noise = rng.uniform(-p, p)
                new_tau = max(0.1, min(1.0, graph.nodes[nid].tau + noise))
                graph.nodes[nid] = graph.nodes[nid].model_copy(
                    update={"tau": new_tau, "sigma": new_tau}
                )

    elif perturbation == "low_conf_pruning":
        # Remove edges/nodes below confidence threshold p (interpreted as threshold)
        threshold = p
        graph.edges = [e for e in graph.edges if e.confidence >= threshold]
        # Remove isolated non-phi nodes
        connected = {e.source for e in graph.edges} | {e.target for e in graph.edges}
        for nid in list(graph.nodes.keys()):
            if nid != "phi" and nid not in connected:
                del graph.nodes[nid]

    out = solver.solve(graph, pred.case_id)
    return out.sigma_phi


def run_robustness(
    predictions: list[ADSSPrediction],
    solver,
    cfg: dict,
    perturbation: str = "arg_removal",
    p_levels: list[float] | None = None,
    seed: int = 42,
) -> list[RobustnessResult]:
    """
    Run robustness experiment for one perturbation type across p_levels.
    perturbation: arg_removal | relation_flip | tau_noise | low_conf_pruning
    """
    if p_levels is None:
        p_levels = [0.0, 0.1, 0.2, 0.3, 0.5]

    rng     = random.Random(seed)
    labelled = [p for p in predictions
                if p.gold_label is not None and p.solver_output is not None]

    results = []
    for p in p_levels:
        golds, preds_dec, shifts = [], [], []
        for pred in labelled:
            orig_sigma = pred.sigma_phi
            new_sigma  = _perturb_and_solve(pred, perturbation, p, solver, cfg, rng)
            from src.qbaf.solver import make_decision
            new_dec_str, _ = make_decision(new_sigma, cfg)
            # Map UNCERTAIN → No for metric computation
            new_dec = new_dec_str if new_dec_str != "UNCERTAIN" else "No"
            golds.append(pred.gold_label)
            preds_dec.append(new_dec)
            shifts.append(abs(new_sigma - orig_sigma))

        results.append(RobustnessResult(
            perturbation=perturbation,
            p_level=p,
            accuracy=_accuracy(golds, preds_dec),
            macro_f1=_macro_f1(golds, preds_dec),
            mean_score_shift=sum(shifts) / len(shifts) if shifts else 0.0,
            n_samples=len(labelled),
        ))
        logger.info(
            f"[robustness/{perturbation} p={p:.2f}] "
            f"Acc={results[-1].accuracy:.3f} F1={results[-1].macro_f1:.3f} "
            f"Δσ={results[-1].mean_score_shift:.4f}"
        )
    return results


def print_robustness(results: list[RobustnessResult]) -> None:
    if not results:
        return
    print(f"\n  Robustness [{results[0].perturbation}]")
    print(f"  {'p':>5}  {'Acc':>6}  {'F1':>6}  {'Δσ':>8}")
    for r in results:
        print(f"  {r.p_level:>5.2f}  {r.accuracy:>6.3f}  "
              f"{r.macro_f1:>6.3f}  {r.mean_score_shift:>8.4f}")
