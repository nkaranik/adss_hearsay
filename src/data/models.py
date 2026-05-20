"""
Pydantic data models for the generic ADSS system.
phi is now dynamic — extracted from the case, not hardcoded.
"""
from __future__ import annotations

import json
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class Stance(str, Enum):
    SUPPORT = "support"
    ATTACK  = "attack"

class RelationType(str, Enum):
    SUPPORT = "support"
    ATTACK  = "attack"
    NEUTRAL = "neutral"

class Decision(str, Enum):
    YES       = "Yes"
    NO        = "No"
    UNCERTAIN = "UNCERTAIN"

class SolverType(str, Enum):
    DF_QUAD      = "df_quad"
    QE_SEMANTICS = "qe_semantics"

class EscalationAction(str, Enum):
    RERUN_EXTRACTION = "rerun_extraction"
    STRONGER_MODEL   = "stronger_model"
    HUMAN_REVIEW     = "human_review"


# ── Dataset ───────────────────────────────────────────────────────────────────

class CaseExample(BaseModel):
    """A single case for analysis — domain-agnostic."""
    case_id:           str
    text:              str
    decision_question: Optional[str] = None   # phi — inferred by LLM if blank
    label:             Optional[str] = None   # gold label (never leaked to prompts)
    split:             str = "demo"

# Keep old name as alias for backward compat
HearsayExample = CaseExample


# ── Argumentation Mining ──────────────────────────────────────────────────────

class ClaimNode(BaseModel):
    id:   str = "phi"
    text: str = ""   # populated dynamically from LLM

class Argument(BaseModel):
    id:              str
    text:            str
    stance_to_claim: Stance
    legal_rule:      str
    evidence_span:   Optional[str] = None
    confidence:      float = Field(ge=0.0, le=1.0)

    @field_validator("id")
    @classmethod
    def id_not_phi(cls, v: str) -> str:
        if v == "phi":
            raise ValueError("'phi' is reserved for the claim node.")
        return v

class Relation(BaseModel):
    source:     str
    target:     str
    type:       RelationType
    confidence: float = Field(ge=0.0, le=1.0)

class ExtractionResult(BaseModel):
    case_id:           str
    input_text:        str
    claim:             ClaimNode = Field(default_factory=ClaimNode)
    arguments:         list[Argument] = Field(default_factory=list)
    relations:         list[Relation] = Field(default_factory=list)
    extraction_model:  str = ""
    raw_llm_response:  str = ""
    parse_error:       Optional[str] = None
    n_raw_arguments:   int = 0    # count before neutral filtering

    @model_validator(mode="after")
    def _unique_ids(self) -> "ExtractionResult":
        ids = [a.id for a in self.arguments]
        if len(ids) != len(set(ids)):
            raise ValueError("Argument ids must be unique.")
        return self

    @model_validator(mode="after")
    def _valid_refs(self) -> "ExtractionResult":
        valid = {a.id for a in self.arguments} | {"phi"}
        for r in self.relations:
            if r.source not in valid:
                raise ValueError(f"Relation source '{r.source}' not found.")
            if r.target not in valid:
                raise ValueError(f"Relation target '{r.target}' not found.")
        return self


# ── Strength Attribution ──────────────────────────────────────────────────────

class RubricScore(BaseModel):
    """Rubric dimensions — accepts both short and full LLM field name variants."""
    legal_relevance:        float = Field(ge=0.0, le=1.0, alias="legal_relevance")
    factual_grounding:      float = Field(ge=0.0, le=1.0, alias="factual_grounding")
    specificity:            float = Field(ge=0.0, le=1.0, alias="specificity")
    logical_coherence:      float = Field(ge=0.0, le=1.0, alias="logical_coherence")
    fre_801c_applicability: float = Field(ge=0.0, le=1.0, alias="fre_801c_applicability")

    model_config = {"populate_by_name": True}

class ArgumentStrength(BaseModel):
    argument_id:   str
    tau:           float = Field(ge=0.1, le=1.0)
    rubric:        RubricScore
    justification: str
    model:         str = ""


# ── QBAF ─────────────────────────────────────────────────────────────────────

class QBAFNode(BaseModel):
    id:       str
    text:     str
    tau:      float = Field(ge=0.0, le=1.0)
    sigma:    float = Field(ge=0.0, le=1.0)
    is_claim: bool  = False

class QBAFEdge(BaseModel):
    source:     str
    target:     str
    type:       RelationType
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)

class QBAFGraph(BaseModel):
    nodes:       dict[str, QBAFNode] = Field(default_factory=dict)
    edges:       list[QBAFEdge]      = Field(default_factory=list)
    solver_type: SolverType          = SolverType.DF_QUAD

    def supporters_of(self, nid: str) -> list[str]:
        return [e.source for e in self.edges
                if e.target == nid and e.type == RelationType.SUPPORT]

    def attackers_of(self, nid: str) -> list[str]:
        return [e.source for e in self.edges
                if e.target == nid and e.type == RelationType.ATTACK]

class SolverOutput(BaseModel):
    case_id:     str
    graph:       QBAFGraph
    sigma_phi:   float = Field(ge=0.0, le=1.0)
    sigma_all:   dict[str, float] = Field(default_factory=dict)
    iterations:  int  = 0
    converged:   bool = True
    solver_type: SolverType = SolverType.DF_QUAD


# ── Pipeline ──────────────────────────────────────────────────────────────────

class UncertaintyFlags(BaseModel):
    is_uncertain:         bool = False
    sigma_phi:            float = 0.0
    threshold_low:        float = 0.45
    threshold_high:       float = 0.55
    escalation_triggered: bool = False
    escalation_actions:   list[EscalationAction] = Field(default_factory=list)
    escalation_log:       list[str]              = Field(default_factory=list)

class HITLIntervention(BaseModel):
    intervention_type:    str
    target_id:            str
    old_value:            Any
    new_value:            Any
    timestamp:            str = ""
    recomputed_sigma_phi: Optional[float]    = None
    recomputed_decision:  Optional[Decision] = None

class ADSSPrediction(BaseModel):
    case_id:    str
    input_text: str
    phi_text:   str = ""          # the decision question / claim
    gold_label: Optional[str] = None

    extraction:    ExtractionResult
    strengths:     list[ArgumentStrength] = Field(default_factory=list)
    solver_output: Optional[SolverOutput] = None

    sigma_phi:   float    = 0.5
    decision:    Decision = Decision.UNCERTAIN
    uncertainty: UncertaintyFlags = Field(default_factory=UncertaintyFlags)

    hitl_interventions: list[HITLIntervention] = Field(default_factory=list)
    pre_hitl_decision:  Optional[Decision]     = None

    def is_correct(self) -> Optional[bool]:
        if self.gold_label is None:
            return None
        pred = self.decision.value if self.decision != Decision.UNCERTAIN else "No"
        return pred == self.gold_label


class BaselinePrediction(BaseModel):
    case_id:       str
    baseline_name: str
    decision:      Decision
    raw_response:  str
    sigma_phi:     Optional[float] = None
    gold_label:    Optional[str]   = None


class ClassMetrics(BaseModel):
    label:     str
    precision: float
    recall:    float
    f1:        float
    support:   int

class EvaluationReport(BaseModel):
    system_name:           str
    accuracy:              float
    macro_f1:              float
    per_class:             list[ClassMetrics]
    confusion_matrix:      list[list[int]]
    confusion_labels:      list[str]
    accuracy_ci:           tuple[float, float]
    macro_f1_ci:           tuple[float, float]
    n_samples:             int
    n_uncertain_escalated: int = 0

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()
