"""
Module D: HITL Contestability Dashboard — Generic Decision Support.
Run:  streamlit run src/dashboard/app.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

st.set_page_config(
    page_title="ADSS – Decision Support",
    page_icon="⚖️",
    layout="wide",
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_DEFAULT_DECISION = "Is the statement described in this narrative hearsay under FRE 801(c)?"

DECISION_PRESETS = {
    "Hearsay (FRE 801c)":       "Is the statement described in this narrative hearsay under FRE 801(c)?",
    "Criminal guilt":           "Is the defendant guilty of the charged offence?",
    "Contract breach":          "Did the defendant breach the contract?",
    "Medical — surgery needed": "Should the patient undergo surgery?",
    "Custom…":                  "",
}


@st.cache_resource
def _base_cfg():
    from src.utils.helpers import load_config
    return load_config()


def _render_graph(graph) -> str:
    try:
        from pyvis.network import Network  # type: ignore
        net = Network(height="420px", width="100%", directed=True,
                      bgcolor="#0e1117", font_color="#ffffff")
        for nid, node in graph.nodes.items():
            color = "#FFD700" if node.is_claim else (
                "#28a745" if node.sigma >= 0.5 else "#dc3545"
            )
            net.add_node(nid,
                         label=f"{nid}\nτ={node.tau:.2f} σ={node.sigma:.2f}",
                         title=node.text, color=color,
                         shape="star" if node.is_claim else "dot", size=20)
        for e in graph.edges:
            net.add_edge(e.source, e.target,
                         color="#00ff88" if e.type.value == "support" else "#ff4444",
                         title=e.type.value, arrows="to", width=2)
        return net.generate_html()
    except ImportError:
        lines = ["**Graph (install pyvis for visual rendering):**"]
        for e in graph.edges:
            lines.append(f"  {e.source} "
                         f"{'→(+)' if e.type.value=='support' else '→(-)'} {e.target}")
        return "\n".join(lines)


def _show_graph(html: str) -> None:
    if html.strip().startswith("<"):
        st.components.v1.html(html, height=450)
    else:
        st.markdown(html)


def _decision_badge(val: str, sigma: float, claim_text: str = "") -> None:
    color = {"Yes": "#28a745", "No": "#dc3545", "UNCERTAIN": "#FFA500"}.get(val, "#888")
    claim_line = (
        f"<p style='text-align:center;font-size:0.85rem;opacity:0.8;margin-top:4px'>"
        f"φ: {claim_text[:80]}{'…' if len(claim_text)>80 else ''}</p>"
    ) if claim_text else ""
    st.markdown(
        f"<div style='background:{color};padding:14px 24px;border-radius:10px;"
        f"text-align:center;font-size:1.6rem;font-weight:bold;color:white;"
        f"letter-spacing:2px'>{val}</div>"
        f"<p style='text-align:center;margin-top:6px'>σ(φ) = {sigma:.4f}</p>"
        f"{claim_line}",
        unsafe_allow_html=True,
    )


def _auto_extract_phi(narrative: str, cfg: dict) -> str:
    """Ask the LLM to identify the central yes/no decision question from the narrative."""
    import src.utils.llm_client as _llm
    prompt = (
        "Read the following case summary and write one yes/no question that captures "
        "the central decision to be made. Examples: "
        "'Is the defendant liable for breach of contract?' or "
        "'Is the plaintiff entitled to compensation?' or "
        "'Is the statement hearsay under FRE 801(c)?'\n\n"
        "Return ONLY the question — no explanation, no preamble, nothing else.\n\n"
        f"Case summary:\n{narrative[:1500]}"
    )
    # Use the full configured token budget (Qwen3 needs room for reasoning + answer).
    backend = cfg.get("backend", "gemini")
    if backend == "lmstudio":
        max_tok = cfg.get("lmstudio", {}).get("max_tokens", 7000)
    else:
        max_tok = min(cfg.get("gemini", {}).get("max_tokens", 4096), 1024)

    try:
        result = _llm.call_llm(prompt, cfg, max_tokens=max_tok)
        logger.info(f"Auto-extract raw response (first 200 chars): {result[:200]!r}")

        if not result or not result.strip():
            logger.warning(
                "Auto-extract returned empty response. "
                "Gemini may have filtered it or the model returned no text. "
                "Falling back to default decision problem."
            )
            return _DEFAULT_DECISION

        # Take first non-empty line that contains a question mark
        for line in result.strip().splitlines():
            q = line.strip().strip('"').strip("'").strip()
            if q and "?" in q:
                return q
        # Fallback: first non-empty line even without ?
        for line in result.strip().splitlines():
            q = line.strip().strip('"').strip("'").strip()
            if len(q) > 10:   # ignore single-word noise
                return q if q.endswith("?") else q + "?"
        return _DEFAULT_DECISION
    except Exception as e:
        logger.error(f"Auto-extract phi failed: {e}")
        return _DEFAULT_DECISION


def main():
    import src.utils.gemini_client as _gc
    import src.utils.llm_client as _llm_mod

    if "cfg" not in st.session_state:
        st.session_state["cfg"] = dict(_base_cfg())
    cfg = st.session_state["cfg"]

    st.title("⚖️ ADSS — Argumentative Decision Support System")
    st.caption("Neuro-Symbolic HITL · Generic Decision Problems · 1 LLM call per case")

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")

        st.subheader("🔌 Backend")
        backend = st.radio(
            "LLM backend", ["gemini", "lmstudio"],
            index=0 if cfg.get("backend", "gemini") == "gemini" else 1,
            horizontal=True,
        )
        cfg["backend"] = backend

        if backend == "gemini":
            st.subheader("🔑 Gemini API Key")
            active_key = _gc.get_active_key()
            st.markdown(f"**Active key:** `{_gc.masked_key(active_key)}`")
            if active_key:
                st.success("Key is set ✅")
            else:
                st.error("No key ❌")
            with st.expander("Paste / change key", expanded=not bool(active_key)):
                pasted = st.text_input("API key", type="password",
                                       placeholder="AIzaSy…", key="key_input")
                if st.button("Apply key", key="apply_key_btn"):
                    if pasted.strip():
                        _gc.set_runtime_key(pasted.strip())
                        st.success(f"Key set: {_gc.masked_key(pasted.strip())}")
                        st.rerun()
                    else:
                        st.warning("Enter a key first.")

            st.subheader("🤖 Gemini model")
            model_opts = ["gemini-2.5-flash", "gemini-2.5-pro",
                          "gemini-2.0-flash", "gemini-1.5-flash"]
            cur    = cfg.get("gemini", {}).get("model", "gemini-2.5-flash")
            chosen = st.selectbox("Model", model_opts,
                                  index=model_opts.index(cur) if cur in model_opts else 0)
            cfg.setdefault("gemini", {})["model"] = chosen
            st.caption(
                "Free tier: 2.5-flash  |  Paid: 2.5-pro\n"
                "Tip: if 2.5-flash returns empty responses, switch to "
                "**gemini-2.0-flash** (not a thinking model → more reliable "
                "for short structured tasks like auto-extract)."
            )

        else:
            st.subheader("🖥️ LM Studio")
            st.info("Make sure LM Studio is running with a model loaded.")
            lscfg = cfg.setdefault("lmstudio", {})
            lscfg["base_url"] = st.text_input(
                "Base URL", value=lscfg.get("base_url", "http://127.0.0.1:1234/v1"))
            lscfg["model"] = st.text_input(
                "Model name", value=lscfg.get("model", "qwen/qwen3.6-35b-a3b"),
                help="Must match exactly the model loaded in LM Studio")

            st.markdown("**🧠 Thinking mode**")
            thinking_choice = st.radio(
                "Qwen3 reasoning",
                ["Off (faster, ~1-2 min)", "On (slower, ~4-6 min, needs 8192 ctx)"],
                index=0 if lscfg.get("disable_thinking", True) else 1,
            )
            lscfg["disable_thinking"] = thinking_choice.startswith("Off")
            if not lscfg["disable_thinking"]:
                st.warning(
                    "⚠️ Thinking ON: Qwen3 may contradict its own reasoning in JSON. "
                    "Thinking OFF recommended. Use context ≥ 8192 if ON."
                )
            st.caption("No API key needed for local models.")

            st.divider()
            st.markdown("**⚙️ Token Budget**")
            ctx_size = st.selectbox(
                "LM Studio context length",
                [4096, 8192, 16384, 32768, 65536, 131072, 262144], index=6)
            recommended = max(1000, ctx_size - 700 - 500)
            lscfg["max_tokens"] = st.slider(
                "max_tokens", 1000, ctx_size - 500,
                min(recommended, lscfg.get("max_tokens", 7000)), 256)
            answer_budget = lscfg["max_tokens"] - 3400
            if answer_budget < 600:
                st.error(f"⚠️ Only ~{answer_budget} tokens for answer. Increase context.")
            elif answer_budget < 1200:
                st.warning(f"~{answer_budget} tokens for answer. May truncate.")
            else:
                st.success(f"~{answer_budget} tokens for answer ✅")

            if st.button("🔗 Test connection"):
                try:
                    from src.utils.lmstudio_client import call_lmstudio
                    r = call_lmstudio(
                        "Reply with exactly one word: OK",
                        model=lscfg["model"], max_tokens=50,
                        base_url=lscfg["base_url"],
                        disable_thinking=lscfg.get("disable_thinking", True),
                    )
                    st.success(f"Connected ✅ — {r.strip()[:40]}")
                except Exception as e:
                    st.error(f"Connection failed: {e}")

        st.divider()
        st.subheader("🧮 Solver")
        solver    = st.selectbox("Semantics", ["df_quad", "qe_semantics"])
        threshold = st.slider("Decision threshold σ(φ)", 0.0, 1.0, 0.5, 0.01)
        ub_lo     = st.slider("Uncertainty band low",  0.0, 0.5, 0.45, 0.01)
        ub_hi     = st.slider("Uncertainty band high", 0.5, 1.0, 0.55, 0.01)
        max_args  = st.slider("Max arguments", 2, 10,
                              cfg.get("mining", {}).get("max_arguments", 6))
        cfg.setdefault("qbaf", {}).update({
            "solver": solver, "decision_threshold": threshold,
            "uncertainty_band": [ub_lo, ub_hi],
        })
        cfg.setdefault("mining", {})["max_arguments"] = max_args
        from src.qbaf.solver import get_solver
        st.caption(f"Backend: **{_llm_mod.backend_label(cfg)}**")

    # ── Input — narrative FIRST, then phi ─────────────────────────────────────
    st.header("1 · Case Input")

    narrative = st.text_area(
        "Case narrative:",
        value=(
            "At trial for fraud, Officer Reyes testified that the victim told him "
            "over the phone: 'I transferred the money as instructed by my advisor.' "
            "The statement is offered to prove the advisor gave the instruction."
        ),
        height=160,
        key="narrative_input",
    )
    case_id = st.text_input("Case ID", value="case_001")

    st.divider()
    st.markdown("**Decision problem (φ) — what yes/no question should the system answer?**")

    preset = st.selectbox(
        "Preset",
        list(DECISION_PRESETS.keys()),
        key="preset_sel",
    )

    # Session-state key ensures value persists across reruns
    if "phi_text" not in st.session_state:
        st.session_state["phi_text"] = DECISION_PRESETS.get(preset, "")

    # When preset changes (and is not Custom), sync the text box
    if preset != "Custom…":
        st.session_state["phi_text"] = DECISION_PRESETS[preset]

    phi_col, btn_col = st.columns([4, 1])
    with phi_col:
        decision_problem = st.text_input(
            "Decision problem:",
            value=st.session_state["phi_text"],
            placeholder="e.g. Is the defendant liable for breach of contract?",
            key="phi_input",
        )
        # Keep session state in sync with manual edits
        st.session_state["phi_text"] = decision_problem

    with btn_col:
        st.markdown("<br>", unsafe_allow_html=True)  # vertical align
        auto_btn = st.button(
            "🤖 Auto-extract",
            help="Let the LLM read the narrative and suggest the decision question.",
            disabled=not narrative.strip(),
            key="auto_phi_btn",
        )

    if auto_btn and narrative.strip():
        with st.spinner("Extracting decision problem from narrative…"):
            suggested = _auto_extract_phi(narrative, cfg)
        st.session_state["phi_text"] = suggested
        st.info(f"Suggested φ: **{suggested}**")
        st.rerun()

    ready = (backend == "lmstudio") or bool(_gc.get_active_key())
    can_run = ready and bool(decision_problem.strip()) and bool(narrative.strip())

    c1, c2 = st.columns([2, 1])
    with c1:
        run = st.button("🔍  Analyse Case  (1 API call)",
                        type="primary", disabled=not can_run)
    with c2:
        if st.button("🗑️  Clear"):
            st.session_state.pop("pred", None)
            st.rerun()

    if not ready:
        st.warning("⬅️ Set your API key or switch to LM Studio in the sidebar.")
    elif not decision_problem.strip():
        st.warning("Enter a decision problem above, or click **🤖 Auto-extract**.")

    if "pred" not in st.session_state:
        st.session_state["pred"] = None

    if run and can_run:
        from src.data.models import HearsayExample
        from src.pipeline.orchestrator import ADSSPipeline

        # Store phi alongside pred so display is always consistent
        st.session_state["last_phi"] = decision_problem.strip()

        ex = HearsayExample(
            case_id=case_id,
            text=narrative,
            split="demo",
            decision_problem=decision_problem.strip(),
        )
        pipeline = ADSSPipeline(cfg)
        pipeline.solver = get_solver(cfg)
        with st.spinner(f"Running via **{_llm_mod.backend_label(cfg)}**…"):
            try:
                st.session_state["pred"] = pipeline.run_case(ex, save_artifacts=False)
            except Exception as e:
                err = str(e)
                if "ContextTooSmallError" in type(e).__name__ or "Context window overflow" in err:
                    st.error(
                        "**Context window too small.**\n\n"
                        "LM Studio: load model → ⚙️ gear → Context Length → 8192 → Apply.\n\n"
                        f"Detail: {err[:200]}"
                    )
                elif "RESOURCE_EXHAUSTED" in err or "429" in err:
                    st.error("**Rate limit (429).** Try another model or wait.")
                elif "503" in err or "UNAVAILABLE" in err:
                    st.error("**Server busy (503).** Try again in a moment.")
                elif "Connection" in err or "refused" in err:
                    st.error("**LM Studio not reachable.** Check it is running.")
                else:
                    st.error(f"Error: {e}")

    pred = st.session_state.get("pred")
    if pred is None:
        return

    # Use stored phi (from when analysis was run) — not current input state
    stored_phi = st.session_state.get("last_phi", decision_problem)
    # Use stored phi when claim text is empty or a generic default
    _generic = {"The central claim", "The statement is hearsay under FRE 801(c)"}
    claim_text = (
        pred.extraction.claim.text
        if pred.extraction.claim and pred.extraction.claim.text not in _generic
        else stored_phi
    )

    # ── About ────────────────────────────────────────────────────────────────
    with st.expander("ℹ️ About this analysis", expanded=False):
        st.markdown(
            f"**Decision problem (φ):**\n> *{claim_text}*\n\n"
            "**How it works:** arguments extracted from the narrative are classified "
            "as *supporting* or *attacking* φ, scored τ ∈ [0.1,1], propagated through "
            "a QBAF graph, and the final dialectical strength σ(φ) determines:\n"
            f"- **Yes** → σ(φ) ≥ {threshold:.2f}\n"
            f"- **No** → σ(φ) < {threshold:.2f}\n"
            f"- **UNCERTAIN** → σ(φ) ∈ [{ub_lo:.2f}, {ub_hi:.2f}]\n\n"
            "*Neutral arguments (irrelevant to φ) are excluded from the graph.*"
        )

    # ── Decision ──────────────────────────────────────────────────────────────
    st.header("2 · Decision")
    d_col, w_col = st.columns([1, 2])
    with d_col:
        _decision_badge(pred.decision.value, pred.sigma_phi, claim_text)
    with w_col:
        if pred.uncertainty.is_uncertain:
            st.warning(
                f"⚠️ **UNCERTAIN** — σ(φ) = {pred.sigma_phi:.4f} ∈ "
                f"[{ub_lo:.2f}, {ub_hi:.2f}]. Human review recommended."
            )
        if pred.pre_hitl_decision:
            st.info(f"Pre-HITL decision: **{pred.pre_hitl_decision.value}**")

    # ── Arguments ─────────────────────────────────────────────────────────────
    st.header("3 · Extracted Arguments")
    n_raw  = pred.extraction.n_raw_arguments
    n_kept = len(pred.extraction.arguments)
    n_neut = max(0, n_raw - n_kept)

    if n_neut > 0 and n_kept == 0:
        st.info(
            f"ℹ️ The model found **{n_raw} argument(s)** but classified all as "
            "**neutral** with respect to the decision problem.\n\n"
            f"Decision problem asked: *{claim_text}*\n\n"
            "This usually means the narrative does not directly address that question. "
            "Try rephrasing φ or clicking **🤖 Auto-extract** to let the LLM suggest "
            "a better question."
        )
    elif n_neut > 0:
        st.info(
            f"ℹ️ {n_neut} of {n_raw} argument(s) were neutral and excluded from the graph."
        )

    if not pred.extraction.arguments:
        if pred.extraction.parse_error:
            st.error(f"Parse error: {pred.extraction.parse_error[:400]}")
        with st.expander("🔍 Raw LLM response", expanded=True):
            raw = pred.extraction.raw_llm_response
            st.text_area("Raw output:", value=raw[:3000] if raw else "(empty)", height=200)
    else:
        sigma_map = pred.solver_output.sigma_all if pred.solver_output else {}
        tau_map   = {s.argument_id: s.tau for s in pred.strengths}
        st.dataframe([{
            "ID": a.id, "Stance": a.stance_to_claim.value,
            "τ": round(tau_map.get(a.id, a.confidence), 3),
            "σ": round(sigma_map.get(a.id, 0.0), 3),
            "Rule": a.legal_rule,
            "Text": (a.text[:90] + "…") if len(a.text) > 90 else a.text,
        } for a in pred.extraction.arguments], use_container_width=True)

    # ── Graph ─────────────────────────────────────────────────────────────────
    st.header("4 · QBAF Graph")
    if pred.solver_output:
        _show_graph(_render_graph(pred.solver_output.graph))

    # ── HITL ──────────────────────────────────────────────────────────────────
    st.header("5 · Human-in-the-Loop Interventions")
    if not pred.solver_output:
        st.info("No solver output — no arguments were extracted.")
        return

    from src.pipeline.orchestrator import ADSSPipeline as _P
    pipeline_hitl        = _P(cfg)
    pipeline_hitl.solver = get_solver(cfg)
    graph   = pred.solver_output.graph
    non_phi = [nid for nid in graph.nodes if nid != "phi"]

    t1, t2, t3 = st.tabs(["Edit τ (strength)", "Edit Edges", "Add Edge"])

    with t1:
        if not non_phi:
            st.info("No argument nodes to edit.")
        else:
            sel  = st.selectbox("Argument node", non_phi, key="tau_sel")
            cur  = graph.nodes[sel].tau
            nval = st.slider(f"New τ for **{sel}**",
                             0.1, 1.0, float(round(cur, 2)), 0.05, key="tau_sl")
            if st.button("Apply τ change"):
                from src.data.models import HITLIntervention
                pred = pipeline_hitl.apply_hitl_intervention(pred, HITLIntervention(
                    intervention_type="edit_tau", target_id=sel,
                    old_value=cur, new_value=nval,
                ))
                st.session_state["pred"] = pred
                st.success(f"τ({sel})={nval:.2f} → σ(φ)={pred.sigma_phi:.4f} → {pred.decision.value}")
                st.rerun()

    with t2:
        edge_opts = [f"{e.source}→{e.target} [{e.type.value}]" for e in graph.edges]
        if not edge_opts:
            st.info("No edges in graph.")
        else:
            sel_e = st.selectbox("Edge", edge_opts, key="edge_sel")
            ekey  = sel_e.split(" ")[0]
            c1, c2 = st.columns(2)
            with c1:
                if st.button("🔀 Flip support↔attack"):
                    from src.data.models import HITLIntervention
                    pred = pipeline_hitl.apply_hitl_intervention(pred, HITLIntervention(
                        intervention_type="flip_edge", target_id=ekey,
                        old_value=None, new_value=None,
                    ))
                    st.session_state["pred"] = pred
                    st.success(f"Flipped → {pred.decision.value}")
                    st.rerun()
            with c2:
                if st.button("🗑️ Delete edge"):
                    from src.data.models import HITLIntervention
                    pred = pipeline_hitl.apply_hitl_intervention(pred, HITLIntervention(
                        intervention_type="delete_edge", target_id=ekey,
                        old_value=None, new_value=None,
                    ))
                    st.session_state["pred"] = pred
                    st.success(f"Deleted → {pred.decision.value}")
                    st.rerun()

    with t3:
        all_ids  = list(graph.nodes.keys())
        new_src  = st.selectbox("Source", all_ids, key="ns")
        new_tgt  = st.selectbox("Target", all_ids, key="nt")
        new_type = st.selectbox("Type", ["support", "attack"], key="ntype")
        if st.button("➕ Add edge"):
            from src.data.models import HITLIntervention
            pred = pipeline_hitl.apply_hitl_intervention(pred, HITLIntervention(
                intervention_type="add_edge",
                target_id=f"{new_src}→{new_tgt}",
                old_value=None,
                new_value={"source": new_src, "target": new_tgt,
                           "type": new_type, "confidence": 1.0},
            ))
            st.session_state["pred"] = pred
            st.success(f"Added {new_src}→{new_tgt} [{new_type}] → {pred.decision.value}")
            st.rerun()

    # ── Intervention log ──────────────────────────────────────────────────────
    st.header("6 · Intervention Log")
    if pred.hitl_interventions:
        st.dataframe([{
            "Type":           i.intervention_type,
            "Target":         i.target_id,
            "Old":            str(i.old_value)[:30],
            "New":            str(i.new_value)[:30],
            "σ(φ) after":    f"{i.recomputed_sigma_phi:.4f}" if i.recomputed_sigma_phi else "—",
            "Decision after": i.recomputed_decision.value if i.recomputed_decision else "—",
            "Time":           i.timestamp[:19],
        } for i in pred.hitl_interventions], use_container_width=True)
    else:
        st.info("No interventions yet.")

    st.divider()
    st.download_button(
        "💾 Export prediction JSON",
        data=pred.model_dump_json(indent=2),
        file_name=f"{pred.case_id}_prediction.json",
        mime="application/json",
    )


if __name__ == "__main__":
    main()
