"""
Module C: Symbolic QBAF Engine.

Two pluggable solvers:
  - DFQuADSolver      (DF-QuAD iterative semantics – default)
  - QESemanticsSolver (Quadratic Energy semantics)

DF-QuAD equations (Baroni et al. 2019)
────────────────────────────────────────
  CS(x) = 1 − ∏_{s∈S(x)} (1 − σ(s))       [0 if S(x) = ∅]
  CA(x) = 1 − ∏_{a∈A(x)} (1 − σ(a))       [0 if A(x) = ∅]
  σ(x)  = clamp( τ(x) + (1−τ(x))·CS(x) − τ(x)·CA(x) )

QE Semantics (energy-normalisation)
────────────────────────────────────
  σ(x) = (τ(x) + Σσ(S(x))) / (1 + Σσ(S(x)) + Σσ(A(x)))

  This keeps σ ∈ (0,1], rises above τ when supporters are present,
  and falls below τ when attackers dominate.

Both iterate until ||σ_new − σ_old||_∞ < ε  (handles cyclic graphs).
"""
from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod

from src.data.models import QBAFGraph, SolverOutput, SolverType

logger = logging.getLogger(__name__)


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))


def _topological_order(graph: QBAFGraph) -> list[str]:
    """Kahn's algorithm; appends any remaining cycle nodes at the end."""
    in_deg: dict[str, int] = {nid: 0 for nid in graph.nodes}
    for e in graph.edges:
        in_deg[e.target] = in_deg.get(e.target, 0) + 1

    queue = sorted(nid for nid, d in in_deg.items() if d == 0)
    order: list[str] = []
    while queue:
        nid = queue.pop(0)
        order.append(nid)
        for e in graph.edges:
            if e.source == nid:
                in_deg[e.target] -= 1
                if in_deg[e.target] == 0:
                    queue.append(e.target)

    for nid in graph.nodes:
        if nid not in order:
            order.append(nid)
    return order


class BaseQBAFSolver(ABC):
    def __init__(self, cfg: dict):
        qcfg           = cfg.get("qbaf", {})
        self.max_iters = qcfg.get("max_iterations", 50)
        self.eps       = qcfg.get("convergence_eps", 1e-6)

    @abstractmethod
    def solve(self, graph: QBAFGraph, case_id: str = "") -> SolverOutput: ...

    def _finalise(
        self,
        graph: QBAFGraph,
        sigma: dict[str, float],
        case_id: str,
        iters: int,
        converged: bool,
    ) -> SolverOutput:
        for nid, s in sigma.items():
            if nid in graph.nodes:
                graph.nodes[nid].sigma = s
        phi_default = graph.nodes["phi"].tau if "phi" in graph.nodes else 0.5
        return SolverOutput(
            case_id=case_id,
            graph=graph,
            sigma_phi=sigma.get("phi", phi_default),
            sigma_all=sigma,
            iterations=iters,
            converged=converged,
            solver_type=graph.solver_type,
        )


class DFQuADSolver(BaseQBAFSolver):
    """
    Iterative DF-QuAD semantics.
    σ(x) = clamp(τ + (1−τ)·CS − τ·CA)
    where CS = 1−∏(1−σ(s)), CA = 1−∏(1−σ(a)).
    """

    def solve(self, graph: QBAFGraph, case_id: str = "") -> SolverOutput:
        sigma = {nid: n.tau for nid, n in graph.nodes.items()}
        order = _topological_order(graph)
        converged = False
        iters = 0

        for iters in range(1, self.max_iters + 1):
            old = dict(sigma)
            for nid in order:
                tau = graph.nodes[nid].tau
                sup = graph.supporters_of(nid)
                att = graph.attackers_of(nid)
                cs  = 1.0 - math.prod(1.0 - sigma.get(s, 0.0) for s in sup) if sup else 0.0
                ca  = 1.0 - math.prod(1.0 - sigma.get(a, 0.0) for a in att) if att else 0.0
                sigma[nid] = _clamp(tau + (1.0 - tau) * cs - tau * ca)

            if max(abs(sigma[k] - old[k]) for k in sigma) < self.eps:
                converged = True
                break

        if not converged:
            logger.warning(f"[{case_id}] DF-QuAD: no convergence after {self.max_iters} iters.")
        return self._finalise(graph, sigma, case_id, iters, converged)


class QESemanticsSolver(BaseQBAFSolver):
    """
    Quadratic Energy (normalisation) semantics.
    σ(x) = (τ(x) + Σσ(S)) / (1 + Σσ(S) + Σσ(A))

    Properties:
      - No supporters/attackers  → σ = τ / 1 = τ  (identity)
      - Pure support             → σ > τ  (support raises)
      - Pure attack              → σ < τ  (attack lowers)
      - σ always in (0, 1]
    """

    def solve(self, graph: QBAFGraph, case_id: str = "") -> SolverOutput:
        sigma = {nid: n.tau for nid, n in graph.nodes.items()}
        order = _topological_order(graph)
        converged = False
        iters = 0

        for iters in range(1, self.max_iters + 1):
            old = dict(sigma)
            for nid in order:
                tau   = graph.nodes[nid].tau
                agg_s = sum(sigma.get(s, 0.0) for s in graph.supporters_of(nid))
                agg_a = sum(sigma.get(a, 0.0) for a in graph.attackers_of(nid))
                sigma[nid] = _clamp((tau + agg_s) / (1.0 + agg_s + agg_a))

            if max(abs(sigma[k] - old[k]) for k in sigma) < self.eps:
                converged = True
                break

        if not converged:
            logger.warning(f"[{case_id}] QE: no convergence after {self.max_iters} iters.")
        return self._finalise(graph, sigma, case_id, iters, converged)


def get_solver(cfg: dict) -> BaseQBAFSolver:
    name = cfg.get("qbaf", {}).get("solver", "df_quad")
    if name == "qe_semantics":
        logger.info("Solver: QE Semantics")
        return QESemanticsSolver(cfg)
    logger.info("Solver: DF-QuAD")
    return DFQuADSolver(cfg)


def make_decision(sigma_phi: float, cfg: dict) -> tuple[str, bool]:
    """Return (label, is_uncertain). Label ∈ {'Yes','No','UNCERTAIN'}."""
    threshold = cfg.get("qbaf", {}).get("decision_threshold", 0.5)
    low, high = cfg.get("qbaf", {}).get("uncertainty_band", [0.45, 0.55])
    if low <= sigma_phi <= high:
        return "UNCERTAIN", True
    return ("Yes" if sigma_phi >= threshold else "No"), False
