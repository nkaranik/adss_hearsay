#!/usr/bin/env python3
"""
Generate 3 illustrative HITL case studies.
Usage:  python scripts/case_studies.py
"""
from __future__ import annotations

import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.helpers import load_config, setup_logging, ensure_api_key
from src.data.models import HearsayExample, HITLIntervention, Decision
from src.pipeline.orchestrator import ADSSPipeline

CASES = [
    {
        "id": "cs_001", "label": "CORRECT_INITIAL", "gold": "Yes",
        "text": (
            "At trial for fraud, the prosecutor introduced testimony from a bank "
            "officer that a customer told him: 'I transferred the money as "
            "instructed by my financial advisor.' Offered to prove the advisor "
            "gave the instruction."
        ),
        "description": "Classic hearsay: out-of-court statement offered for truth.",
    },
    {
        "id": "cs_002", "label": "CORRECTED_BY_HITL", "gold": "No",
        "text": (
            "In a breach of contract case the plaintiff's attorney offered the "
            "defendant's own signed letter admitting: 'I failed to deliver goods "
            "by the agreed date.' The defendant objected the letter is hearsay."
        ),
        "description": "Party-opponent admission (FRE 801(d)(2)): NOT hearsay. HITL corrects initial error.",
    },
    {
        "id": "cs_003", "label": "UNCERTAINTY_ESCALATION", "gold": "Yes",
        "text": (
            "A witness overheard two unidentified persons near a storage unit "
            "saying 'the package contains drugs.' Offered to prove the package "
            "did contain drugs. The identity of the declarants is unclear."
        ),
        "description": "Borderline case triggering uncertainty escalation.",
    },
]


def run(pipeline: ADSSPipeline, cs: dict) -> dict:
    pred = pipeline.run_case(
        HearsayExample(case_id=cs["id"], text=cs["text"],
                       label=cs["gold"], split="case_study"),
        save_artifacts=True,
    )
    result = {
        "case_id": cs["id"], "label": cs["label"], "gold": cs["gold"],
        "description": cs["description"],
        "initial_sigma_phi": pred.sigma_phi,
        "initial_decision": pred.decision.value,
        "is_correct_initial": pred.is_correct(),
        "n_args": len(pred.extraction.arguments),
        "is_uncertain": pred.uncertainty.is_uncertain,
        "escalation_log": pred.uncertainty.escalation_log,
    }

    if cs["label"] == "CORRECTED_BY_HITL":
        attack_args = [a for a in pred.extraction.arguments
                       if a.stance_to_claim.value == "attack"]
        if attack_args and pred.solver_output:
            tgt     = attack_args[0].id
            old_tau = pred.solver_output.graph.nodes[tgt].tau
            pred    = pipeline.apply_hitl_intervention(pred, HITLIntervention(
                intervention_type="edit_tau", target_id=tgt,
                old_value=old_tau, new_value=0.95,
            ))
            result["hitl"] = {
                "target": tgt, "old_tau": old_tau, "new_tau": 0.95,
                "new_sigma_phi": pred.sigma_phi,
                "new_decision": pred.decision.value,
                "decision_changed": pred.decision.value != result["initial_decision"],
                "correct_after": pred.is_correct(),
            }

    # Build plain-text explanation
    lines = [
        f"{'='*60}",
        f"Case {cs['id']} — {cs['label']}",
        f"Gold: {cs['gold']}  |  {cs['description']}",
        "",
        f"Step 1 – Extraction: {result['n_args']} arguments",
    ]
    if pred.solver_output:
        for a in pred.extraction.arguments:
            t = pred.solver_output.graph.nodes[a.id].tau
            s = pred.solver_output.sigma_all.get(a.id, 0.0)
            lines.append(f"  [{a.id}] {a.stance_to_claim.value.upper()} "
                         f"τ={t:.2f} σ={s:.2f}  {a.text[:60]}…")
    lines += [
        "",
        f"Step 2 – Initial σ(φ)={result['initial_sigma_phi']:.4f}",
        f"Step 3 – Initial decision: {result['initial_decision']} "
        f"(correct={result['is_correct_initial']})",
    ]
    if result["is_uncertain"]:
        lines.append(f"Step 4 – UNCERTAIN zone → {result['escalation_log']}")
    if "hitl" in result:
        h = result["hitl"]
        lines += [
            "",
            f"Step 5 – HITL: boost τ({h['target']}) {h['old_tau']:.2f}→{h['new_tau']:.2f}",
            f"Step 6 – New σ(φ)={h['new_sigma_phi']:.4f} → {h['new_decision']}",
            f"Step 7 – Decision changed: {h['decision_changed']} | Correct: {h['correct_after']}",
        ]
    lines.append(f"\nFinal: {pred.decision.value}  σ(φ)={pred.sigma_phi:.4f}")
    result["explanation"] = "\n".join(lines)
    print(result["explanation"])
    return result


def main():
    cfg = load_config()
    setup_logging(cfg)
    ensure_api_key()

    pipeline = ADSSPipeline(cfg)
    out_dir  = Path(cfg.get("system", {}).get("artifact_dir", "artifacts")) / "case_studies"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = [run(pipeline, cs) for cs in CASES]
    (out_dir / "all_case_studies.json").write_text(
        json.dumps(all_results, indent=2, default=str, encoding="utf-8")
    )
    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    main()
