"""Unit tests for QBAF solvers (no LLM calls)."""
import pytest
from src.data.models import QBAFGraph, QBAFEdge, QBAFNode, RelationType
from src.qbaf.solver import DFQuADSolver, QESemanticsSolver

_CFG = {"qbaf": {"max_iterations": 100, "convergence_eps": 1e-8,
                  "decision_threshold": 0.5, "uncertainty_band": [0.45, 0.55]}}


def _simple():
    return QBAFGraph(
        nodes={
            "a1":  QBAFNode(id="a1",  text="s", tau=0.8, sigma=0.8),
            "a2":  QBAFNode(id="a2",  text="a", tau=0.3, sigma=0.3),
            "phi": QBAFNode(id="phi", text="c", tau=0.5, sigma=0.5, is_claim=True),
        },
        edges=[
            QBAFEdge(source="a1", target="phi", type=RelationType.SUPPORT, confidence=1.0),
            QBAFEdge(source="a2", target="phi", type=RelationType.ATTACK,  confidence=1.0),
        ],
    )


def _solo():
    return QBAFGraph(
        nodes={"phi": QBAFNode(id="phi", text="c", tau=0.5, sigma=0.5, is_claim=True)},
        edges=[],
    )


class TestDFQuAD:
    def test_no_edges_sigma_equals_tau(self):
        assert abs(DFQuADSolver(_CFG).solve(_solo()).sigma_phi - 0.5) < 1e-6

    def test_support_increases_sigma(self):
        g = QBAFGraph(
            nodes={"a1": QBAFNode(id="a1", text="s", tau=0.9, sigma=0.9),
                   "phi": QBAFNode(id="phi", text="c", tau=0.5, sigma=0.5, is_claim=True)},
            edges=[QBAFEdge(source="a1", target="phi", type=RelationType.SUPPORT, confidence=1.0)],
        )
        assert DFQuADSolver(_CFG).solve(g).sigma_phi > 0.5

    def test_attack_decreases_sigma(self):
        g = QBAFGraph(
            nodes={"a1": QBAFNode(id="a1", text="a", tau=0.9, sigma=0.9),
                   "phi": QBAFNode(id="phi", text="c", tau=0.5, sigma=0.5, is_claim=True)},
            edges=[QBAFEdge(source="a1", target="phi", type=RelationType.ATTACK, confidence=1.0)],
        )
        assert DFQuADSolver(_CFG).solve(g).sigma_phi < 0.5

    def test_sigma_in_unit_interval(self):
        out = DFQuADSolver(_CFG).solve(_simple())
        assert all(0.0 <= v <= 1.0 for v in out.sigma_all.values())

    def test_converged(self):
        assert DFQuADSolver(_CFG).solve(_simple()).converged


class TestQESemantics:
    def test_no_edges_sigma_equals_tau(self):
        assert abs(QESemanticsSolver(_CFG).solve(_solo()).sigma_phi - 0.5) < 1e-6

    def test_support_increases_sigma(self):
        g = QBAFGraph(
            nodes={"a1": QBAFNode(id="a1", text="s", tau=0.9, sigma=0.9),
                   "phi": QBAFNode(id="phi", text="c", tau=0.5, sigma=0.5, is_claim=True)},
            edges=[QBAFEdge(source="a1", target="phi", type=RelationType.SUPPORT, confidence=1.0)],
        )
        assert QESemanticsSolver(_CFG).solve(g).sigma_phi > 0.5

    def test_sigma_in_unit_interval(self):
        out = QESemanticsSolver(_CFG).solve(_simple())
        assert all(0.0 <= v <= 1.0 for v in out.sigma_all.values())
