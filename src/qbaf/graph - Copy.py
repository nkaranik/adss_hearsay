"""QBAF Graph construction — unchanged logic, updated import."""
from __future__ import annotations
import logging
from src.data.models import (
    ArgumentStrength, ExtractionResult, QBAFEdge, QBAFGraph,
    QBAFNode, RelationType, SolverType, Stance,
)
logger = logging.getLogger(__name__)

def build_qbaf(
    extraction:      ExtractionResult,
    strengths:       list[ArgumentStrength],
    phi_tau:         float      = 0.5,
    solver_type:     SolverType = SolverType.DF_QUAD,
    include_neutral: bool       = False,
) -> QBAFGraph:
    strength_map = {s.argument_id: s.tau for s in strengths}
    nodes: dict[str, QBAFNode] = {
        "phi": QBAFNode(
            id="phi", text=extraction.claim.text,
            tau=phi_tau, sigma=phi_tau, is_claim=True,
        )
    }
    for arg in extraction.arguments:
        tau = max(0.1, min(1.0, strength_map.get(arg.id, arg.confidence)))
        nodes[arg.id] = QBAFNode(id=arg.id, text=arg.text, tau=tau, sigma=tau)

    edges: list[QBAFEdge] = []
    for rel in extraction.relations:
        if rel.type == RelationType.NEUTRAL and not include_neutral:
            continue
        if rel.source not in nodes or rel.target not in nodes:
            continue
        edges.append(QBAFEdge(
            source=rel.source, target=rel.target,
            type=rel.type, confidence=rel.confidence,
        ))

    args_with_phi = {e.source for e in edges if e.target == "phi"}
    for arg in extraction.arguments:
        if arg.id not in args_with_phi:
            etype = (RelationType.SUPPORT
                     if arg.stance_to_claim == Stance.SUPPORT
                     else RelationType.ATTACK)
            edges.append(QBAFEdge(
                source=arg.id, target="phi",
                type=etype, confidence=arg.confidence,
            ))

    return QBAFGraph(nodes=nodes, edges=edges, solver_type=solver_type)
