"QBAF Graph construction."
from __future__ import annotations

import logging

from src.data.models import (
    ArgumentStrength,
    ExtractionResult,
    QBAFEdge,
    QBAFGraph,
    QBAFNode,
    RelationType,
    SolverType,
    Stance,
)

logger = logging.getLogger(__name__)


def build_qbaf(
    extraction: ExtractionResult,
    strengths: list[ArgumentStrength],
    phi_tau: float = 0.5,
    solver_type: SolverType = SolverType.DF_QUAD,
    include_neutral: bool = False,
) -> QBAFGraph:
    """Build a QBAF graph from an extraction result.

    Important fallback rule:
    - Keep all LLM-provided valid relations.
    - Add a fallback direct edge to ``phi`` only for arguments that have no
      outgoing relation at all.

    This prevents double-counting undercutters/rebuttals. For example, if the
    extractor returns ``C1 -> A8`` as an attack, ``C1`` should not be added again
    as ``C1 -> phi`` unless the LLM explicitly provided that direct relation.
    """
    strength_map = {s.argument_id: s.tau for s in strengths}

    nodes: dict[str, QBAFNode] = {
        "phi": QBAFNode(
            id="phi",
            text=extraction.claim.text,
            tau=phi_tau,
            sigma=phi_tau,
            is_claim=True,
        )
    }

    for arg in extraction.arguments:
        tau = max(0.1, min(1.0, strength_map.get(arg.id, arg.confidence)))
        nodes[arg.id] = QBAFNode(
            id=arg.id,
            text=arg.text,
            tau=tau,
            sigma=tau,
        )

    edges: list[QBAFEdge] = []
    seen_edges: set[tuple[str, str, RelationType]] = set()

    for rel in extraction.relations:
        if rel.type == RelationType.NEUTRAL and not include_neutral:
            continue
        if rel.source not in nodes or rel.target not in nodes:
            continue

        key = (rel.source, rel.target, rel.type)
        if key in seen_edges:
            continue
        seen_edges.add(key)

        edges.append(QBAFEdge(
            source=rel.source,
            target=rel.target,
            type=rel.type,
            confidence=rel.confidence,
        ))

    # Fallback direct-to-phi edges are useful when the LLM omits relations.
    # However, do not add fallback phi edges for arguments that already have an
    # outgoing argument-to-argument relation. Those arguments are usually
    # undercutters/rebuttals whose effect is represented through their target.
    args_with_direct_phi = {e.source for e in edges if e.target == "phi"}
    args_with_any_outgoing = {e.source for e in edges}

    for arg in extraction.arguments:
        if arg.id in args_with_direct_phi:
            continue
        if arg.id in args_with_any_outgoing:
            logger.debug(
                "Not auto-adding %s -> phi: argument already has outgoing relation(s).",
                arg.id,
            )
            continue

        etype = (
            RelationType.SUPPORT
            if arg.stance_to_claim == Stance.SUPPORT
            else RelationType.ATTACK
        )
        edges.append(QBAFEdge(
            source=arg.id,
            target="phi",
            type=etype,
            confidence=arg.confidence,
        ))

    return QBAFGraph(nodes=nodes, edges=edges, solver_type=solver_type)
