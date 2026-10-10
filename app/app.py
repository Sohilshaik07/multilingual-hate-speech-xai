"""
app/app.py

Streamlit Research Demonstration Interface
Project: Faithfulness-Aware Explainability Audit of Multilingual Transformer Models for Code-Mixed Hate Speech Detection
Team: Team 6

Critical Research Constraints:
1. Operates exclusively on validated inference checkpoints (HECM, Tamil, Malayalam).
2. Authoritative historical research results are immutable and displayed as fixed baselines.
3. No training or metric manufacturing is performed in this UI.
4. Explanations adhere to behavioral relevance framing; no claims of causal faithfulness.
"""

import os
import sys
from pathlib import Path
import streamlit as st
import pandas as pd

# Ensure repository root is in Python module search path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from app.utils.labels import (
        LABEL_MAPPINGS,
        get_labels_for_dataset,
        get_label_name,
        get_samples_for_dataset,
    )
    from app.utils.model_loader import (
        load_cached_pipeline,
        predict_sample,
        get_checkpoint_storage_info,
        get_model_source,
        get_model_identifier,
        HF_REPO_IDS,
    )
    from app.components.prediction import render_prediction_panel
    from app.components.explanations import (
        compute_shap_word_scores,
        compute_attention_word_scores,
        render_shap_panel,
        render_attention_panel,
        render_side_by_side_panel,
        render_scientific_disclaimer,
    )
except (ModuleNotFoundError, ImportError):
    from utils.labels import (
        LABEL_MAPPINGS,
        get_labels_for_dataset,
        get_label_name,
        get_samples_for_dataset,
    )
    from utils.model_loader import (
        load_cached_pipeline,
        predict_sample,
        get_checkpoint_storage_info,
        get_model_source,
        get_model_identifier,
        HF_REPO_IDS,
    )
    from components.prediction import render_prediction_panel
    from components.explanations import (
        compute_shap_word_scores,
        compute_attention_word_scores,
        render_shap_panel,
        render_attention_panel,
        render_side_by_side_panel,
        render_scientific_disclaimer,
    )


# --- Page Configuration ---
st.set_page_config(
    page_title="Team 6 | Explainability Audit",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar navigation and theme control. These only affect presentation.
st.sidebar.markdown("## ✳ Team 6 XAI Lab")
st.sidebar.caption("Faithfulness-aware explainability audit")
st.sidebar.markdown("---")
selected_page = st.sidebar.radio(
    "WORKSPACE",
    ["Analysis workbench", "Audit findings", "Calibration & robustness", "Methodology", "Project resources"],
    index=0,
    key="active_workspace_page",
)
st.sidebar.markdown("---")
dark_mode = st.sidebar.toggle("Dark theme", value=True, key="dark_mode")
st.sidebar.caption("Validated mBERT checkpoints · Fixed research baselines")

# --- Design system: editorial research workspace ---
_theme = {
    "bg": "#0b1220" if dark_mode else "#f4f7fb",
    "panel": "#111d2e" if dark_mode else "#ffffff",
    "panel2": "#16263a" if dark_mode else "#eef3f8",
    "ink": "#edf4ff" if dark_mode else "#17263b",
    "muted": "#9aaec4" if dark_mode else "#68788d",
    "line": "#26384d" if dark_mode else "#dce4ed",
    "field": "#0e1929" if dark_mode else "#fbfdff",
    "accent": "#2dd4bf",
}
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
    :root {{ --bg:{_theme['bg']}; --panel:{_theme['panel']}; --panel2:{_theme['panel2']}; --ink:{_theme['ink']}; --muted:{_theme['muted']}; --line:{_theme['line']}; --field:{_theme['field']}; --accent:{_theme['accent']}; }}
    html, body, [class*="css"] {{ font-family:'DM Sans',sans-serif; }}
    .stApp {{ background:var(--bg); color:var(--ink); }}
    [data-testid="stHeader"] {{ background:var(--bg); }}
    [data-testid="stSidebar"] {{ background:var(--panel); border-right:1px solid var(--line); }}
    [data-testid="stSidebar"] * {{ color:var(--ink); }}
    .block-container {{ max-width:1480px; padding-top:2.25rem; padding-bottom:3rem; }}
    h1,h2,h3,h4 {{ font-family:'Manrope',sans-serif; color:var(--ink) !important; letter-spacing:-.03em; }}
    p, li, label, .stMarkdown {{ color:var(--ink); }}
    [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * {{ color:var(--muted) !important; }}
    /* Header: deliberately simple, with enough top clearance for Streamlit's toolbar */
    .topline {{ display:flex; justify-content:space-between; align-items:center; gap:1rem; margin:0 0 1.45rem; padding:.35rem 0 .8rem; border-bottom:1px solid var(--line); }}
    .brand {{ display:flex; align-items:center; gap:.7rem; color:var(--ink); font-weight:800; font-size:.9rem; line-height:1.35; min-width:0; }}
    .brand-mark {{ display:inline-flex; flex:0 0 36px; width:36px; height:36px; align-items:center; justify-content:center; border:1px solid #2b485d; border-radius:11px; background:#14263a; color:#5eead4; font-size:1.05rem; }}
    .brand-name {{ display:block; color:var(--ink); }}
    .brand-sub {{ color:var(--muted); font-weight:500; }}
    .topmeta {{ color:var(--muted); font-size:.67rem; font-weight:750; letter-spacing:.12em; text-transform:uppercase; text-align:right; white-space:nowrap; }}
    .intro {{ margin:0 0 1.35rem; padding-top:.15rem; }}
    .eyebrow,.section-eyebrow {{ color:var(--accent); text-transform:uppercase; letter-spacing:.14em; font-size:.68rem; font-weight:800; margin:0 0 .55rem; line-height:1.5; }}
    .intro-title {{ font-family:'Manrope',sans-serif; font-size:clamp(1.9rem,3.2vw,2.65rem); line-height:1.14; font-weight:800; letter-spacing:-.05em; margin:0 0 .7rem; color:var(--ink); overflow-wrap:anywhere; }}
    .intro-copy {{ color:var(--muted); max-width:850px; line-height:1.7; font-size:.97rem; margin:0; }}
    .status-strip {{ display:flex; flex-wrap:wrap; gap:.5rem; margin-top:1.1rem; }}
    .status-pill {{ border:1px solid var(--line); background:var(--panel); color:var(--muted); border-radius:999px; padding:.4rem .72rem; font-size:.73rem; font-weight:650; }}
    @media(max-width:640px) {{ .topline {{ align-items:flex-start; }} .topmeta {{ max-width:120px; white-space:normal; line-height:1.4; }} .brand {{ font-size:.82rem; }} }}
    [data-testid="stVerticalBlockBorderWrapper"] {{ background:var(--panel); border-color:var(--line); border-radius:17px; }}
    .context-card {{ background:linear-gradient(145deg,#132438,#102f3d); color:#edf6fa; border:1px solid #294257; border-radius:18px; padding:1.35rem; min-height:100%; box-shadow:0 16px 35px rgba(0,0,0,.12); }}
    .context-card h3 {{ color:#fff !important; font-size:1.12rem; margin:.35rem 0 .6rem; }}
    .context-card p {{ color:#c4d2dc; font-size:.86rem; line-height:1.55; }}
    .context-label {{ color:#5eead4; font-size:.67rem; font-weight:800; letter-spacing:.13em; text-transform:uppercase; }}
    .context-stat {{ border-top:1px solid rgba(255,255,255,.14); padding:.72rem 0 .15rem; margin-top:.6rem; }}
    .context-stat span {{ display:block; color:#a8bac7; font-size:.74rem; margin-bottom:.2rem; }}
    .context-stat strong {{ color:#f4f8fb; font-size:.9rem; font-weight:700; overflow-wrap:anywhere; }}
    .context-foot,.quiet-note {{ color:var(--muted) !important; font-size:.77rem !important; line-height:1.5; }}
    [data-testid="stMetric"] {{ background:var(--panel2); border:1px solid var(--line); padding:.9rem; border-radius:13px; }}
    [data-testid="stMetricLabel"], [data-testid="stMetricDelta"] {{ color:var(--muted) !important; }}
    [data-testid="stMetricValue"] {{ color:var(--ink) !important; font-weight:800; letter-spacing:-.035em; }}
    [data-testid="stTextArea"] textarea, [data-testid="stSelectbox"] > div, [data-testid="stMultiSelect"] > div {{ border-radius:10px; }}
    [data-testid="stTextArea"] textarea, [data-testid="stSelectbox"] div[data-baseweb="select"] > div {{ background:var(--field); color:var(--ink); border-color:var(--line); }}
    [data-testid="stExpander"] {{ border-color:var(--line); border-radius:12px; background:var(--panel); }}
    [data-testid="stDataFrame"] {{ border:1px solid var(--line); border-radius:12px; overflow:hidden; }}
    [data-testid="stAlert"] {{ border-radius:12px; }}
    div.stButton > button[kind="primary"] {{ border:0; border-radius:10px; font-weight:750; min-height:2.8rem; background:#0f9f9a; color:white; box-shadow:0 5px 15px rgba(15,159,154,.18); }}
    div.stButton > button[kind="primary"]:hover {{ background:#0b827f; border:0; }}
    hr {{ border-color:var(--line); }}
    [data-testid="stRadio"] label {{ border-radius:9px; }}
    @media(max-width:900px) {{ .block-container {{padding:1rem 1rem 2rem;}} }}
    </style>
    """, unsafe_allow_html=True,
)

# --- Compact, unclipped product header ---
st.markdown(
    """
    <header class="topline">
      <div class="brand">
        <span class="brand-mark" aria-hidden="true">✳</span>
        <span>
          <span class="brand-name">TEAM 6 <span class="brand-sub">/ XAI RESEARCH LAB</span></span>
        </span>
      </div>
      <div class="topmeta">MULTILINGUAL MODEL AUDIT</div>
    </header>
    <section class="intro">
      <div class="eyebrow">Faithfulness-aware explainability audit</div>
      <h1 class="intro-title">Look inside the prediction.</h1>
      <p class="intro-copy">Explore how a multilingual mBERT classifier responds to code-mixed text, then compare SHAP attributions with final-layer self-attention. Research benchmarks remain fixed to the validated evaluation.</p>
      <div class="status-strip">
        <span class="status-pill">3 language settings</span>
        <span class="status-pill">Validated mBERT checkpoints</span>
        <span class="status-pill">SHAP + self-attention</span>
        <span class="status-pill">Fixed research baselines</span>
      </div>
    </section>
    """, unsafe_allow_html=True,
)

LANG_OPTIONS = {
    "Hindi-English (HECM)": "hecm",
    "Tamil-English (DravidianCodeMix)": "tamil",
    "Malayalam-English (DravidianCodeMix)": "malayalam",
}

ckpt_specs = {
    "hecm": ("results/modeling/mbert_hecm/checkpoint-824", "Binary (2 classes)"),
    "tamil": ("results/modeling/mbert_tamil/checkpoint-3844", "Multiclass (6 classes)"),
    "malayalam": ("results/modeling/mbert_malayalam/checkpoint-1756", "Multiclass (5 classes)"),
}

if selected_page == "Analysis workbench":
    # --- Primary workbench: input is the focus, context is secondary ---
    left, right = st.columns([1.72, 1], gap="large")
    with left:
        with st.container(border=True):
            st.markdown('<div class="section-eyebrow">01 / Interactive workbench</div>', unsafe_allow_html=True)
            st.markdown("### Analyze a text sample")
            st.caption("Choose a language, load a preset, or paste your own code-mixed text.")
            selected_lang_name = st.selectbox(
                "Language pair & model", options=list(LANG_OPTIONS.keys()), index=0,
                help="Switches the active checkpoint, label mapping, and benchmark context.",
            )
            dataset_key = LANG_OPTIONS[selected_lang_name]
            samples = get_samples_for_dataset(dataset_key)
            sample_options = ["Custom text input..."] + [sample["title"] for sample in samples]
            def _load_selected_preset() -> None:
                chosen = st.session_state.get("selected_sample_title", "Custom text input...")
                st.session_state["user_input_text"] = next(
                    (sample["text"] for sample in samples if sample["title"] == chosen), ""
                )

            selected_sample_title = st.selectbox(
                "Benchmark sample preset (optional)", options=sample_options, index=1 if samples else 0,
                key="selected_sample_title", on_change=_load_selected_preset,
            )
            if "user_input_text" not in st.session_state:
                st.session_state["user_input_text"] = next(
                    (sample["text"] for sample in samples if sample["title"] == selected_sample_title), ""
                )
            user_input = st.text_area(
                "Text to analyze", height=145, key="user_input_text",
                placeholder="Paste a social post or code-mixed commentary…",
                help="Supports Latin, Devanagari, Tamil, and Malayalam scripts.",
            )
            button_col, helper_col = st.columns([1, 2.1], vertical_alignment="center")
            with button_col:
                analyze_button = st.button("✦  Run analysis", type="primary", use_container_width=True)
            with helper_col:
                st.markdown('<div class="quiet-note">Inference and local explanation signals run on the selected validated checkpoint. Aggregate audit metrics are not recomputed.</div>', unsafe_allow_html=True)

    with right:
        ckpt_rel, class_desc = ckpt_specs[dataset_key]
        _model_source = get_model_source()
        _model_id = HF_REPO_IDS.get(dataset_key, "(unknown)") if _model_source == "hf" else ckpt_rel
        _source_label = "Hugging Face Hub" if _model_source == "hf" else "Local checkpoint"
        st.markdown(
            f"""
            <div class="context-card">
              <div class="context-label">02 / Model context</div>
              <h3>{selected_lang_name}</h3>
              <p>Validated multilingual transformer checkpoint selected for this analysis.</p>
              <div class="context-stat"><span>Checkpoint source</span><strong>{_source_label}</strong></div>
              <div class="context-stat"><span>Model repository / path</span><strong>{_model_id}</strong></div>
              <div class="context-stat"><span>Output space</span><strong>{class_desc}</strong></div>
              <div class="context-stat"><span>Architecture</span><strong>mBERT · 128-token window</strong></div>
              <p class="context-foot">Frozen checkpoints · seed 42 · 73,286 benchmark instances across the study.</p>
            </div>
            """, unsafe_allow_html=True,
        )

    # Load only the selected model. The loader/cache behavior is unchanged.
    try:
        model, tokenizer, device = load_cached_pipeline(dataset_key)
        st.caption(f"● Model ready on {device.upper()} · { _source_label }")
    except Exception as e:
        st.error(f"Could not load the selected checkpoint: {e}")
        st.stop()

    st.markdown('<div class="quiet-note">Research demonstration only. Predictions are model outputs, not ground-truth labels or definitive judgments. Aggregate audit figures remain fixed to the authoritative evaluation.</div>', unsafe_allow_html=True)
    st.markdown("---")

    # --- Execution State ---
    if "last_result" not in st.session_state or st.session_state.get("last_dataset") != dataset_key:
        st.session_state["last_result"] = None
        st.session_state["shap_scores"] = None
        st.session_state["attention_scores"] = None
        st.session_state["last_analyzed_input"] = None
        st.session_state["last_dataset"] = dataset_key

    if analyze_button:
        if not user_input or not user_input.strip():
            st.warning("⚠️ Please enter valid non-empty text before analyzing.")
            st.session_state["last_result"] = None
            st.session_state["shap_scores"] = None
            st.session_state["attention_scores"] = None
            st.session_state["last_analyzed_input"] = None
        else:
            pred_res = None
            with st.spinner("Executing mBERT inference..."):
                try:
                    pred_res = predict_sample(model, tokenizer, user_input, dataset_key, device=device)
                    st.session_state["last_result"] = pred_res
                    st.session_state["last_analyzed_input"] = user_input.strip()
                except Exception as e:
                    st.error(f"⚠️ Inference failed: {e}")
                    st.session_state["last_result"] = None
                    st.session_state["shap_scores"] = None
                    st.session_state["attention_scores"] = None
                    st.session_state["last_analyzed_input"] = None

            if pred_res is not None:
                with st.spinner("Computing Word-level Self-Attention and SHAP attributions..."):
                    visible_text = pred_res["visible_text"]
                    target_class = pred_res["predicted_class"]

                    # Compute Attention
                    try:
                        attn_scores = compute_attention_word_scores(
                            model=model,
                            tokenizer=tokenizer,
                            visible_text=visible_text,
                            device=device
                        )
                    except Exception as e:
                        attn_scores = []
                    st.session_state["attention_scores"] = attn_scores

                    # Compute SHAP
                    try:
                        shap_scores = compute_shap_word_scores(
                            model=model,
                            tokenizer=tokenizer,
                            visible_text=visible_text,
                            target_class=target_class,
                            device=device
                        )
                    except Exception as e:
                        shap_scores = []
                    st.session_state["shap_scores"] = shap_scores


# --- Main content selected from the sidebar ---
st.markdown("---")


# =========================================================================
# TAB 1: Prediction & Explanations
# =========================================================================
if selected_page == "Analysis workbench":
    if st.session_state["last_result"] is not None:
        pred_res = st.session_state["last_result"]
        shap_scores = st.session_state["shap_scores"]
        attn_scores = st.session_state["attention_scores"]
        target_class_str = get_label_name(dataset_key, pred_res["predicted_class"])

        last_analyzed = st.session_state.get("last_analyzed_input", "").strip()
        current_input = user_input.strip()
        if last_analyzed and current_input and last_analyzed != current_input:
            st.warning("⚠️ **Input Modified:** The text in the input box differs from the analyzed text below. Click **[ 🚀 Analyze Text ]** to evaluate your updated text.")

        # 1. Prediction panel
        render_prediction_panel(pred_res, dataset_key)

        st.markdown("---")
        # 2. Side-by-side comparison
        render_side_by_side_panel(shap_scores, attn_scores)

        st.markdown("---")
        # 3. Detailed SHAP
        render_shap_panel(shap_scores, target_class_str)

        st.markdown("---")
        # 4. Detailed Attention
        render_attention_panel(attn_scores)

        st.markdown("---")
        # 5. Scientific disclaimer
        render_scientific_disclaimer()
    else:
        st.info("Run an analysis above to see the prediction, probability distribution, and explanation signals here.")


# =========================================================================
# TAB 2: Validated Audit Findings
# =========================================================================
if selected_page == "Audit findings":
    st.markdown("### Validated Audit Findings")
    st.caption("Aggregate results from the fixed research audit conducted under Google Colab GPU acceleration.")

    st.markdown("#### Benchmark snapshot")
    metric_cols = st.columns(3, gap="medium")
    benchmark_cards = [
        ("Hindi-English · HECM", "81.47%", "Macro-F1 0.7696"),
        ("Tamil-English", "71.01%", "Macro-F1 0.4694"),
        ("Malayalam-English", "91.89%", "Macro-F1 0.4504"),
    ]
    for metric_col, (language, accuracy, macro_f1) in zip(metric_cols, benchmark_cards):
        with metric_col:
            with st.container(border=True):
                st.caption(language)
                st.markdown(f"<div style='font-size:1.75rem;font-weight:800;letter-spacing:-.04em;color:#2dd4bf'>{accuracy}</div>", unsafe_allow_html=True)
                st.caption(macro_f1 + " · fixed benchmark")

    st.markdown("#### 1. Predictive Performance & Faithfulness Audit Summary")
    
    audit_summary_df = pd.DataFrame([
        {
            "Language Pair": "HECM (Hindi-English)",
            "Accuracy": "0.8147",
            "Macro-F1": "0.7696",
            "SHAP Top vs Random p": "0.001134 (Sig)",
            "Attention Top vs Random p": "0.156692 (Non-sig)",
            "Top-Word Agreement": "23.0%",
            "Mean Spearman ρ": "0.3128"
        },
        {
            "Language Pair": "Tamil-English",
            "Accuracy": "0.7101",
            "Macro-F1": "0.4694",
            "SHAP Top vs Random p": "0.000220 (Sig)",
            "Attention Top vs Random p": "0.620854 (Non-sig)",
            "Top-Word Agreement": "18.0%",
            "Mean Spearman ρ": "0.1858"
        },
        {
            "Language Pair": "Malayalam-English",
            "Accuracy": "0.9189",
            "Macro-F1": "0.4504",
            "SHAP Top vs Random p": "0.000245 (Sig)",
            "Attention Top vs Random p": "0.079719 (Marginal)",
            "Top-Word Agreement": "26.0%",
            "Mean Spearman ρ": "0.1677"
        }
    ])
    st.dataframe(audit_summary_df, use_container_width=True, hide_index=True)

    st.markdown("#### 2. Key Epistemic Findings")
    st.markdown(
        r"""
        - **Accuracy Masking:** In class-imbalanced datasets (Tamil 72% non-offensive; Malayalam 88% non-offensive), 
          high raw accuracy conceals severe minority-class performance collapse (Macro-F1 drops to 0.45–0.47).
        - **SHAP Behavioral Relevance:** KernelSHAP token deletions consistently disrupt model decisions significantly 
          more than random deletions ($p < 0.001$ across all three language settings).
        - **Attention Behavioral Relevance:** Attention did not show statistically significant behavioral relevance in the top-vs-random deletion comparison in any evaluated language setting. This finding is limited to the tested protocol and does not establish that attention can never be useful.
        - **Cross-Method Disagreement:** Top-1 agreement between SHAP and Attention remains low ($\le 26\%$), with mean 
          Spearman rank correlations of only 0.17–0.31, proving that the two methods prioritize distinct input tokens.
        """
    )


# =========================================================================
# TAB 3: Calibration & Robustness
# =========================================================================
if selected_page == "Calibration & robustness":
    st.markdown("### Model-Level Calibration & Robustness")
    st.caption("Authoritative evaluation of confidence reliability and input perturbations.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Confidence Calibration (15 Uniform Bins)")
        st.markdown(
            r"""
            *Model-level calibration results from the research evaluation:*
            - **HECM:** $\text{ECE} = 0.071284$ | $\text{Brier} = 0.282710$ | Mean Conf: 88.2% (Acc: 81.5%, Gap: **+6.7%**)
            - **Tamil:** $\text{ECE} = 0.066790$ | $\text{Brier} = 0.393545$ | Mean Conf: 77.7% (Acc: 71.0%, Gap: **+6.7%**)
            - **Malayalam:** $\text{ECE} = 0.065863$ | $\text{Brier} = 0.146696$ | Mean Conf: 98.4% (Acc: 91.9%, Gap: **+6.5%**)
            
            **Systematic Overconfidence:** Across all three languages, model posterior confidence exceeds empirical 
            accuracy by approximately 6.5–6.8 percentage points.
            """
        )

    with col2:
        st.markdown("#### Perturbation & Script Robustness")
        st.markdown(
            r"""
            *Perturbation experimental findings:*
            - **Surface Formatting Robustness:**
              - Evaluated across 8 distinct typographic experiments (whitespace collapsing, punctuation spacing).
              - **Prediction Consistency:** 100.0% (Flip Rate: 0.0%).
              - **Mean Absolute Confidence Shift:** $< 2 \times 10^{-7}$.
            - **Unicode Script Composition Degradation:**
              - **HECM:** Macro-F1 drops from 0.772 (Single-script) to 0.675 (Mixed-script) (**-12.6%**).
              - **Tamil:** Macro-F1 drops from 0.468 to 0.452 (**-3.3%**).
              - **Malayalam:** Macro-F1 drops from 0.446 to 0.336 (**-24.8%**).
            """
        )


# =========================================================================
# TAB 4: Model & Methodology
# =========================================================================
if selected_page == "Methodology":
    st.markdown("### Model Architecture & Audit Methodology")
    
    st.markdown(
        r"""
        - **Foundation Model:** `bert-base-multilingual-cased` (mBERT) pre-trained across 104 languages.
        - **Fine-Tuning Setup:** Conducted in Google Colab with AdamW optimizer, linear learning rate warmup, sequence length 128.
        - **Explainability Audit:**
          - **KernelSHAP:** WordPiece token attributions with zero-embedding baseline on $N=100$ fixed instances per language.
          - **Self-Attention:** Multi-head attention averaged across all 12 heads from the final encoder layer (Layer 12).
          - **Deletion Protocol:** Measuring confidence drops and decision flips under Top-word, Bottom-word, and Random-word deletions.
          - **Hypothesis Testing:** One-sided Wilcoxon signed-rank test ($\alpha = 0.05$) testing whether Top deletion causes larger confidence drops than Random deletion.
        """
    )
    
    with st.expander("Methodological Limitations (L1 – L9)"):
        st.markdown(
            """
            - **L1 (Single Architecture):** Evaluates mBERT only; comparative models (XLM-R, IndicBERT) are outside final test scope.
            - **L2 (Audit Sample Size):** KernelSHAP evaluated on 100 instances per language due to computational cost.
            - **L3 (Deletion Protocol):** Deletion introduces out-of-distribution input syntax.
            - **L4 (Script Proxy):** Unicode block analysis treats Latin transliteration as single-script.
            - **L5 (Binning Sensitivity):** ECE computed with uniform $M=15$ bins.
            - **L6 (Narrow Perturbation):** Formatting tests limited to typographic spacing.
            - **L7 (Uncalibrated):** Raw posteriors evaluated; post-hoc calibration reserved for deployment.
            - **L8 (Conflicting Labels):** Natural dataset noise preserved.
            - **L9 (Attribution Families):** KernelSHAP is sole local perturbation method evaluated.
            """
        )


# =========================================================================
# TAB 5: Project Resources
# =========================================================================
if selected_page == "Project resources":
    st.markdown("### Project Resources & Deliverables")

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.markdown("#### Authoritative Reports")
        st.markdown(
            """
            - **Academic Research Report (PDF):** `reports/final_report.pdf` (16 pages)
            - **Academic Research Report (Markdown):** `reports/final_report.md`
            - **Report Validation Certificate:** `reports/REPORT_VALIDATION.md`
            - **Phase 6 Final QA Certificate:** `reports/PHASE6_FINAL_QA.md`
            """
        )

    with col_r2:
        st.markdown("#### Checkpoint Footprint & Deployment")
        storage = get_checkpoint_storage_info()
        active_src = storage.get("model_source", "hf")
        if active_src == "hf":
            src_note = (
                "☁️ **Active source: HuggingFace Hub** — weights are streamed on first inference. "
                "No local storage required for this deployment."
            )
        else:
            src_note = (
                f"💾 **Active source: Local** — total checkpoint size "
                f"**{storage['total_size_gb']} GB** ({storage['total_size_mb']} MB). "
                "Set `MODEL_SOURCE=hf` to switch to HuggingFace Hub streaming."
            )
        st.markdown(
            f"""
            - **HECM:** `sohilshaik07/team6-mbert-hecm` ({storage['hecm']['size_mb']} MB local)
            - **Tamil:** `sohilshaik07/team6-mbert-tamil` ({storage['tamil']['size_mb']} MB local)
            - **Malayalam:** `sohilshaik07/team6-mbert-malayalam` ({storage['malayalam']['size_mb']} MB local)

            {src_note}
            """
        )

    st.markdown("---")
    st.markdown("#### Publication Figures Browser")
    
    fig_stems = [
        ("fig01_predictive_performance", "Fig 1: Predictive Performance Divergence"),
        ("fig02_calibration_ece", "Fig 2: Calibration ECE Summary"),
        ("fig03_reliability_hecm", "Fig 3: HECM Reliability Diagram"),
        ("fig04_reliability_tamil", "Fig 4: Tamil Reliability Diagram"),
        ("fig05_reliability_malayalam", "Fig 5: Malayalam Reliability Diagram"),
        ("fig06_shap_faithfulness", "Fig 6: KernelSHAP Deletion Faithfulness"),
        ("fig07_attention_faithfulness", "Fig 7: Self-Attention Faithfulness Evaluation"),
        ("fig08_attention_shap_agreement", "Fig 8: Attention-SHAP Cross-Method Agreement"),
        ("fig09_formatting_robustness", "Fig 9: Surface Formatting Robustness"),
        ("fig10_script_composition", "Fig 10: Unicode Script Composition Degradation"),
    ]
    
    selected_fig = st.selectbox(
        "Choose a publication figure to inspect:",
        options=[f[1] for f in fig_stems]
    )
    
    for stem, title in fig_stems:
        if title == selected_fig:
            fig_path = REPO_ROOT / f"figures/{stem}.png"
            if fig_path.exists():
                st.image(str(fig_path), caption=f"{title} (300 DPI Publication Figure)", use_container_width=True)
            else:
                st.warning(f"Figure file not found at: {fig_path}")
            break
