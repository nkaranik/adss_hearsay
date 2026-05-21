"""
End-to-end ADSS pipeline orchestrator — generic decision support.
Supports any decision problem, not just FRE hearsay.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.data.models import (
    ADSSPrediction, BaselinePrediction, Decision,
    EscalationAction, ExtractionResult, HearsayExample,
    HITLIntervention, QBAFEdge, RelationType,
    SolverType, UncertaintyFlags,
)
from src.mining.extractor import ArgumentExtractor, _DEFAULT_DECISION
from src.scoring.strength import StrengthAttributor
from src.qbaf.graph import build_qbaf
from src.qbaf.solver import get_solver, make_decision
import src.utils.llm_client as _llm

logger = logging.getLogger(__name__)


def _uncertainty_flags(sigma_phi: float, cfg: dict, case_id: str) -> UncertaintyFlags:
    low, high    = cfg.get("qbaf", {}).get("uncertainty_band", [0.45, 0.55])
    is_uncertain = low <= sigma_phi <= high
    flags = UncertaintyFlags(
        is_uncertain=is_uncertain, sigma_phi=sigma_phi,
        threshold_low=low, threshold_high=high,
    )
    if is_uncertain:
        flags.escalation_triggered = True
        flags.escalation_actions   = [
            EscalationAction.RERUN_EXTRACTION,
            EscalationAction.HUMAN_REVIEW,
        ]
        msg = (f"[{case_id}] UNCERTAIN σ(φ)={sigma_phi:.4f} ∈ [{low},{high}]. "
               "Human review recommended.")
        flags.escalation_log.append(msg)
        logger.warning(msg)
    return flags


class ADSSPipeline:
    """Generic ADSS pipeline — one LLM call per case for any decision problem."""

    def __init__(self, cfg: dict):
        self.cfg          = cfg
        self.extractor    = ArgumentExtractor(cfg)
        self.attributor   = StrengthAttributor(cfg)
        self.solver       = get_solver(cfg)
        self.artifact_dir = Path(cfg.get("system", {}).get("artifact_dir", "artifacts"))
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"ADSSPipeline ready. Backend: {_llm.backend_label(cfg)}")

    def run_case(
        self,
        example: HearsayExample,
        save_artifacts: bool = True,
    ) -> ADSSPrediction:
        cid              = example.case_id
        decision_problem = getattr(example, "decision_problem", _DEFAULT_DECISION)
        logger.info(f"=== {cid} === (1 LLM call)")

        # Combined extraction + strength scoring
        extraction, prefilled_strengths = self.extractor.extract_combined(
            cid, example.text, decision_problem
        )
        logger.info(
            f"  {len(extraction.arguments)} args extracted, "
            f"{len(prefilled_strengths)} strengths."
        )

        strengths   = self.attributor.score_all(extraction, prefilled_strengths)
        solver_type = SolverType(self.cfg.get("qbaf", {}).get("solver", "df_quad"))
        phi_tau     = self.cfg.get("qbaf", {}).get("phi_initial_strength", 0.5)
        graph       = build_qbaf(extraction, strengths, phi_tau, solver_type)
        solver_out  = self.solver.solve(graph, cid)
        sigma_phi   = solver_out.sigma_phi

        dec_str, _  = make_decision(sigma_phi, self.cfg)
        uncertainty = _uncertainty_flags(sigma_phi, self.cfg, cid)
        decision    = Decision(dec_str)

        pred = ADSSPrediction(
            case_id=cid, input_text=example.text,
            gold_label=example.label,
            extraction=extraction, strengths=strengths,
            solver_output=solver_out,
            sigma_phi=sigma_phi, decision=decision,
            uncertainty=uncertainty,
        )
        if save_artifacts:
            self._save(pred)
        return pred

    def run_batch(
        self,
        examples: list[HearsayExample],
        save_artifacts: bool = True,
    ) -> list[ADSSPrediction]:
        results: list[ADSSPrediction] = []
        for ex in examples:
            try:
                results.append(self.run_case(ex, save_artifacts))
            except Exception as e:
                logger.error(f"[{ex.case_id}] Pipeline error: {e}")
                results.append(ADSSPrediction(
                    case_id=ex.case_id, input_text=ex.text,
                    gold_label=ex.label,
                    extraction=ExtractionResult(
                        case_id=ex.case_id, input_text=ex.text,
                        parse_error=str(e),
                    ),
                    sigma_phi=0.5, decision=Decision.UNCERTAIN,
                ))
        return results

    def _save(self, pred: ADSSPrediction) -> None:
        (self.artifact_dir / f"{pred.case_id}_prediction.json").write_text(
            pred.model_dump_json(indent=2, encoding="utf-8"), encoding="utf-8"
        )

    def apply_hitl_intervention(
        self,
        pred: ADSSPrediction,
        intervention: HITLIntervention,
    ) -> ADSSPrediction:
        intervention.timestamp = datetime.now(timezone.utc).isoformat()
        pred.pre_hitl_decision = pred.pre_hitl_decision or pred.decision

        if pred.solver_output is None:
            logger.warning("No solver output for HITL.")
            return pred

        graph = pred.solver_output.graph.model_copy(deep=True)
        itype = intervention.intervention_type
        tid   = intervention.target_id

        if itype == "edit_tau":
            if tid in graph.nodes:
                intervention.old_value = graph.nodes[tid].tau
                graph.nodes[tid].tau   = max(0.1, min(1.0, float(intervention.new_value)))
                graph.nodes[tid].sigma = graph.nodes[tid].tau

        elif itype == "flip_edge":
            src, tgt = (tid.split("→") + [""])[:2]
            for edge in graph.edges:
                if edge.source == src and edge.target == tgt:
                    intervention.old_value = edge.type.value
                    edge.type = (
                        RelationType.ATTACK
                        if edge.type == RelationType.SUPPORT
                        else RelationType.SUPPORT
                    )
                    break

        elif itype == "delete_edge":
            src, tgt = (tid.split("→") + [""])[:2]
            before = len(graph.edges)
            graph.edges = [e for e in graph.edges
                           if not (e.source == src and e.target == tgt)]
            intervention.old_value = before - len(graph.edges)

        elif itype == "add_edge":
            try:
                graph.edges.append(QBAFEdge(**dict(intervention.new_value)))
            except Exception as e:
                logger.error(f"HITL add_edge error: {e}")

        new_out = self.solver.solve(graph, pred.case_id)
        new_sig = new_out.sigma_phi
        new_dec = Decision(make_decision(new_sig, self.cfg)[0])

        intervention.recomputed_sigma_phi = new_sig
        intervention.recomputed_decision  = new_dec

        pred.solver_output = new_out
        pred.sigma_phi     = new_sig
        pred.decision      = new_dec
        pred.uncertainty   = _uncertainty_flags(new_sig, self.cfg, pred.case_id)
        pred.hitl_interventions.append(intervention)
        return pred


class BaselineRunner:
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def _call(self, prompt: str) -> str:
        return _llm.call_llm(prompt, self.cfg, max_tokens=1024)

    @staticmethod
    def _extract_decision(text: str) -> Decision:
        import re
        m = re.search(r"Answer:\s*(Yes|No)", text, re.IGNORECASE)
        if m:
            return Decision.YES if m.group(1).lower() == "yes" else Decision.NO
        low = text.lower()
        if "not hearsay" in low or "no hearsay" in low:
            return Decision.NO
        if "hearsay" in low:
            return Decision.YES
        return Decision.NO

    def run_cot(self, example: HearsayExample) -> BaselinePrediction:
        tmpl   = Path("prompts/cot_baseline.txt")
        prompt = (
            tmpl.read_text().replace("{narrative}", example.text)
            if tmpl.exists()
            else f"Analyse this case. End with 'Answer: Yes' or 'Answer: No'.\n\n{example.text}"
        )
        raw = self._call(prompt)
        return BaselinePrediction(
            case_id=example.case_id, baseline_name="zero_shot_cot",
            decision=self._extract_decision(raw),
            raw_response=raw, gold_label=example.label,
        )

    def run_few_shot(self, example: HearsayExample) -> BaselinePrediction:
        ex_path = Path("prompts/few_shot_examples.json")
        shots   = ""
        if ex_path.exists():
            n = self.cfg.get("baselines", {}).get("few_shot_examples", 3)
            for ex in json.loads(ex_path.read_text())[:n]:
                shots += (f"\nNarrative: {ex['narrative']}\n"
                          f"Reasoning: {ex['reasoning']}\n---\n")
        prompt = (
            f"End with exactly 'Answer: Yes' or 'Answer: No'.\n\n"
            f"{shots}\nNarrative: {example.text}\n"
        )
        raw = self._call(prompt)
        return BaselinePrediction(
            case_id=example.case_id, baseline_name="few_shot",
            decision=self._extract_decision(raw),
            raw_response=raw, gold_label=example.label,
        )

    def run_adss_no_symbolic(
        self, extraction: ExtractionResult, strengths: list
    ) -> tuple[float, Decision]:
        from src.data.models import Stance
        if not strengths:
            return 0.5, Decision.UNCERTAIN
        arg_map = {a.id: a for a in extraction.arguments}
        sup = [s.tau for s in strengths
               if arg_map.get(s.argument_id) and
               arg_map[s.argument_id].stance_to_claim == Stance.SUPPORT]
        att = [s.tau for s in strengths
               if arg_map.get(s.argument_id) and
               arg_map[s.argument_id].stance_to_claim == Stance.ATTACK]
        agg_s = sum(sup) / len(sup) if sup else 0.0
        agg_a = sum(att) / len(att) if att else 0.0
        sigma = agg_s / (agg_s + agg_a + 1e-8)
        return max(0.0, min(1.0, sigma)), Decision(make_decision(sigma, self.cfg)[0])
