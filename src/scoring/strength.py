"""Module B: pass-through for combined call strengths."""
from __future__ import annotations
import logging
from src.data.models import ArgumentStrength, ExtractionResult, RubricScore
logger = logging.getLogger(__name__)

_WEIGHTS = {
    "legal_relevance":        0.30,
    "factual_grounding":      0.20,
    "specificity":            0.15,
    "logical_coherence":      0.15,
    "fre_801c_applicability": 0.20,
}

class StrengthAttributor:
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def score_all(
        self,
        extraction: ExtractionResult,
        prefilled:  list[ArgumentStrength] | None = None,
    ) -> list[ArgumentStrength]:
        if prefilled:
            return prefilled
        logger.info("No prefilled strengths; using confidence fallback.")
        results = []
        for arg in extraction.arguments:
            tau = max(0.1, min(1.0, arg.confidence))
            results.append(ArgumentStrength(
                argument_id=arg.id, tau=tau,
                rubric=RubricScore(**{d: tau for d in _WEIGHTS}),
                justification="Fallback from extraction confidence.",
                model="fallback",
            ))
        return results
