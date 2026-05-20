"""Integration tests — single combined LLM call, mocked at module level."""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from src.data.models import HearsayExample, Decision

_COMBINED = json.dumps({
    "case_id": "t1",
    "arguments": [
        {
            "id": "a1", "text": "Out-of-court statement.",
            "stance_to_claim": "support", "legal_rule": "FRE 801(c)(1)",
            "evidence_span": None, "confidence": 0.9,
            "rubric": {"legal_relevance": 0.9, "factual_grounding": 0.8,
                       "specificity": 0.7, "logical_coherence": 0.9,
                       "fre_801c_applicability": 0.9},
            "tau": 0.85, "justification": "Strong argument.",
        },
        {
            "id": "a2", "text": "Offered for truth of matter asserted.",
            "stance_to_claim": "support", "legal_rule": "FRE 801(c)(2)",
            "evidence_span": None, "confidence": 0.85,
            "rubric": {"legal_relevance": 0.85, "factual_grounding": 0.75,
                       "specificity": 0.7, "logical_coherence": 0.85,
                       "fre_801c_applicability": 0.85},
            "tau": 0.80, "justification": "Good argument.",
        },
    ],
    "relations": [
        {"source": "a1", "target": "phi", "type": "support", "confidence": 0.9},
        {"source": "a2", "target": "phi", "type": "support", "confidence": 0.85},
    ],
})

_CFG = {
    "backend": "gemini",
    "gemini":  {"model": "gemini-2.5-flash", "max_tokens": 4096, "temperature": 0.0},
    "mining":  {"max_arguments": 6, "min_confidence": 0.3,
                "include_neutral_relations": False, "max_retries": 1},
    "qbaf":    {"solver": "df_quad", "max_iterations": 50,
                "convergence_eps": 1e-6, "decision_threshold": 0.5,
                "uncertainty_band": [0.45, 0.55], "phi_initial_strength": 0.5},
    "system":  {"artifact_dir": "/tmp/adss_test"},
}

# Patch at the module-level reference used by each module
_EXT  = "src.mining.extractor._llm.call_llm"
_ORCH = "src.pipeline.orchestrator._llm.call_llm"


def test_full_pipeline_single_call():
    """ONE mock call → valid prediction with strengths."""
    call_count = 0
    def _mock(*a, **kw):
        nonlocal call_count
        call_count += 1
        return _COMBINED

    from src.pipeline.orchestrator import ADSSPipeline
    with patch(_EXT, side_effect=_mock):
        pred = ADSSPipeline(_CFG).run_case(
            HearsayExample(case_id="t1",
                           text="Witness said victim told him 'car was red'.",
                           split="test"),
            save_artifacts=False,
        )

    assert call_count == 1, f"Expected 1 LLM call, got {call_count}"
    assert pred.case_id == "t1"
    assert pred.decision in Decision
    assert 0.0 <= pred.sigma_phi <= 1.0
    assert len(pred.extraction.arguments) == 2
    assert len(pred.strengths) == 2
    assert pred.strengths[0].tau > 0.1  # tau computed from rubric weights


def test_hitl_edit_tau():
    from src.pipeline.orchestrator import ADSSPipeline
    from src.data.models import HITLIntervention

    with patch(_EXT, return_value=_COMBINED):
        pipeline = ADSSPipeline(_CFG)
        pred = pipeline.run_case(
            HearsayExample(case_id="t1", text="Witness test.", split="test"),
            save_artifacts=False,
        )

    assert "a1" in pred.solver_output.graph.nodes
    old_sigma = pred.sigma_phi

    pred2 = pipeline.apply_hitl_intervention(pred, HITLIntervention(
        intervention_type="edit_tau", target_id="a1",
        old_value=0.85, new_value=0.1,
    ))
    assert pred2.sigma_phi != old_sigma
    assert len(pred2.hitl_interventions) == 1


def test_baseline_cot():
    from src.pipeline.orchestrator import BaselineRunner
    cot_response = "The statement is out-of-court and offered for truth.\nAnswer: Yes"

    with patch(_ORCH, return_value=cot_response):
        result = BaselineRunner(_CFG).run_cot(
            HearsayExample(case_id="b1", text="test", split="test", label="Yes")
        )
    assert result.decision == Decision.YES


def test_lmstudio_backend_routes_correctly():
    cfg_ls = dict(_CFG, backend="lmstudio",
                  lmstudio={"model": "qwen/qwen3-30b-a3b",
                             "base_url": "http://127.0.0.1:1234/v1",
                             "max_tokens": 4096, "temperature": 0.0})
    import src.utils.lmstudio_client as _ls
    with patch.object(_ls, "call_lmstudio", return_value=_COMBINED) as mock_ls:
        from src.pipeline.orchestrator import ADSSPipeline
        pred = ADSSPipeline(cfg_ls).run_case(
            HearsayExample(case_id="t2", text="test", split="test"),
            save_artifacts=False,
        )
        assert mock_ls.called
    assert len(pred.extraction.arguments) == 2
