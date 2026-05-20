# ADSS-Hearsay
**Neuro-Symbolic Argumentative Decision Support System for LegalBench Hearsay (FRE 801(c))**
Powered by **Google Gemini**.

---

## Quick start

```bash
# 1. Clone & create venv
python -m venv .venv && source .venv/bin/activate

# 2. Install
pip install -r requirements.txt

# 3. Set your Gemini API key (free at https://aistudio.google.com/app/apikey)
export GEMINI_API_KEY=your_key_here

# 4. Launch the HITL dashboard
streamlit run src/dashboard/app.py

# 5. Run evaluation (test split, 20 samples)
python scripts/run_eval.py --split test --max 20

# 6. Run case studies
python scripts/case_studies.py

# 7. Run unit tests (no API key needed)
pytest tests/ -v
```

---

## Architecture

```
Input narrative
      │
      ▼
┌──────────────────────┐
│ Module A: Extraction │  Gemini → JSON arguments + relations
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Module B: Strength   │  Gemini → τ(αᵢ) ∈ [0.1, 1.0]
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Module C: QBAF       │  DF-QuAD or QE → σ(φ) ∈ [0,1]
└──────────┬───────────┘
           │
    ┌──────┴──────┐
    ▼             ▼
 Yes / No      UNCERTAIN → escalation log
           │
           ▼
┌──────────────────────┐
│ Module D: HITL       │  Streamlit dashboard
└──────────────────────┘
```

---

## Solvers

| Solver | Equation |
|--------|----------|
| **DF-QuAD** (default) | `σ(x) = τ(x) + (1−τ)·CS(x) − τ·CA(x)` where `CS = 1−∏(1−σ(s))` |
| **QE Semantics** | `σ(x) = τ(x)·(1+Σσ(S)) / (1+Σσ(S)+Σσ(A))` |

Switch via `configs/config.yaml` → `qbaf.solver: qe_semantics`

---

## Configuration (`configs/config.yaml`)

| Key | Default | Description |
|-----|---------|-------------|
| `gemini.model` | `gemini-2.0-flash` | Gemini model for all LLM calls |
| `qbaf.solver` | `df_quad` | `df_quad` or `qe_semantics` |
| `qbaf.decision_threshold` | `0.5` | σ(φ) cut-off |
| `qbaf.uncertainty_band` | `[0.45, 0.55]` | UAE escalation zone |
| `mining.max_arguments` | `8` | Max args per case |

---

## Output artefacts

```
artifacts/
├── <case_id>_prediction.json
├── case_studies/
│   ├── cs_001.json  (correct initial)
│   ├── cs_002.json  (HITL correction)
│   └── cs_003.json  (uncertainty escalation)
└── evaluation/
    ├── full_adss_report.json
    ├── full_adss_predictions.csv
    └── summary.csv
```
