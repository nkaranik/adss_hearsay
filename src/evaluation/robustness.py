
"""
Robustness and perturbation testing for ADSS argument graphs.

Perturbations:
1. Random argument removal
2. Random support/attack relation flipping
3. Random tau noise

Metric:
- Score stability Delta_p = mean |sigma_original - sigma_perturbed|
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import csv
import json
import random

from src.data.models import RelationType
from src.qbaf.solver import make_decision
from src.evaluation.metrics import accuracy_score, macro_f1_score


@dataclass
class RobustnessRecord:
    perturbation: str
    p: float
    n: int
    score_stability_delta_p: float
    accuracy: float
    macro_f1: float


def _binary(label: str) -> str:
    return "No" if label == "UNCERTAIN" else label


def _decision_from_sigma(sigma: float, cfg: dict) -> str:
    return _binary(make_decision(sigma, cfg)[0])


def _arg_ids(graph: Any) -> list[str]:
    return [nid for nid, node in graph.nodes.items() if nid != "phi" and not getattr(node, "is_claim", False)]


def perturb_graph(graph: Any, perturbation: str, p: float, rng: random.Random, tau_noise_scale: float = 0.15) -> Any:
    g = graph.model_copy(deep=True)
    p = max(0.0, min(1.0, p))

    if perturbation == "argument_removal":
        ids = _arg_ids(g)
        rng.shuffle(ids)
        k = int(round(p * len(ids)))
        remove = set(ids[:k])
        for aid in remove:
            g.nodes.pop(aid, None)
        g.edges = [e for e in g.edges if e.source not in remove and e.target not in remove]

    elif perturbation == "relation_flip":
        edges = list(g.edges)
        rng.shuffle(edges)
        k = int(round(p * len(edges)))
        for e in edges[:k]:
            if e.type == RelationType.SUPPORT:
                e.type = RelationType.ATTACK
            elif e.type == RelationType.ATTACK:
                e.type = RelationType.SUPPORT

    elif perturbation == "tau_noise":
        for aid in _arg_ids(g):
            if rng.random() <= p:
                node = g.nodes[aid]
                noise = rng.uniform(-tau_noise_scale, tau_noise_scale)
                node.tau = max(0.1, min(1.0, float(node.tau) + noise))
                node.sigma = node.tau
    else:
        raise ValueError(f"Unknown perturbation: {perturbation}")

    return g


def run_robustness_suite(
    predictions: list[Any],
    solver: Any,
    cfg: dict,
    perturbations: tuple[str, ...] = ("argument_removal", "relation_flip", "tau_noise"),
    p_levels: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.5),
    seed: int = 42,
) -> list[RobustnessRecord]:
    results: list[RobustnessRecord] = []
    usable = [p for p in predictions if getattr(p, "solver_output", None) is not None and getattr(p, "gold_label", None) in ("Yes", "No")]

    for pert in perturbations:
        for p_level in p_levels:
            rng = random.Random(seed + hash((pert, p_level)) % 1000000)
            golds, preds = [], []
            shifts = []
            for pred in usable:
                graph_p = perturb_graph(pred.solver_output.graph, pert, p_level, rng)
                out = solver.solve(graph_p, pred.case_id)
                sigma_p = out.sigma_phi
                shifts.append(abs(float(pred.sigma_phi) - float(sigma_p)))
                golds.append(pred.gold_label)
                preds.append(_decision_from_sigma(sigma_p, cfg))
            results.append(RobustnessRecord(
                perturbation=pert,
                p=p_level,
                n=len(golds),
                score_stability_delta_p=sum(shifts) / len(shifts) if shifts else 0.0,
                accuracy=accuracy_score(golds, preds),
                macro_f1=macro_f1_score(golds, preds),
            ))
    return results


def write_robustness_report(results: list[RobustnessRecord], out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "robustness_results.json").write_text(json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8")
    with (out / "robustness_results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["perturbation", "p", "n", "score_stability_delta_p", "accuracy", "macro_f1"])
        w.writeheader()
        for r in results:
            w.writerow(asdict(r))
