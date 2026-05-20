"""Unit tests for Pydantic data models."""
import pytest
from pydantic import ValidationError
from src.data.models import (
    Argument, ExtractionResult, QBAFEdge, QBAFGraph, QBAFNode,
    Relation, RelationType, RubricScore, Stance,
)


def test_argument_valid():
    a = Argument(id="a1", text="Out-of-court statement.",
                 stance_to_claim=Stance.SUPPORT,
                 legal_rule="FRE 801(c)", confidence=0.9)
    assert a.id == "a1"


def test_argument_phi_reserved():
    with pytest.raises(ValidationError):
        Argument(id="phi", text="x", stance_to_claim=Stance.SUPPORT,
                 legal_rule="FRE 801(c)", confidence=0.5)


def test_extraction_duplicate_ids():
    with pytest.raises(ValidationError):
        ExtractionResult(
            case_id="x", input_text="y",
            arguments=[
                Argument(id="a1", text="x", stance_to_claim=Stance.SUPPORT,
                         legal_rule="r", confidence=0.5),
                Argument(id="a1", text="y", stance_to_claim=Stance.ATTACK,
                         legal_rule="r", confidence=0.5),
            ],
        )


def test_extraction_invalid_relation_ref():
    with pytest.raises(ValidationError):
        ExtractionResult(
            case_id="x", input_text="y",
            arguments=[
                Argument(id="a1", text="x", stance_to_claim=Stance.SUPPORT,
                         legal_rule="r", confidence=0.5),
            ],
            relations=[
                Relation(source="a99", target="a1",
                         type=RelationType.SUPPORT, confidence=0.8),
            ],
        )


def test_qbaf_graph_helpers():
    g = QBAFGraph(
        nodes={
            "a1":  QBAFNode(id="a1",  text="t", tau=0.8, sigma=0.8),
            "phi": QBAFNode(id="phi", text="c", tau=0.5, sigma=0.5, is_claim=True),
        },
        edges=[QBAFEdge(source="a1", target="phi",
                        type=RelationType.SUPPORT, confidence=0.9)],
    )
    assert g.supporters_of("phi") == ["a1"]
    assert g.attackers_of("phi")  == []


def test_rubric_out_of_range():
    with pytest.raises(ValidationError):
        RubricScore(legal_relevance=1.5, factual_grounding=0.5,
                    specificity=0.5, logical_coherence=0.5,
                    fre_801c_applicability=0.5)
