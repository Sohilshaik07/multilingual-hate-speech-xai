"""
app/components/explanations.py

Explainability calculation and presentation components:
1. SHAP Token/Word Attributions (Kernel / Partition Explainer)
2. Attention-based Token/Word Importance (Layer 12, Multi-head average)
3. Side-by-Side Method Comparison & Disagreement Analysis
4. Scientific Disclaimers & Epistemic Boundaries
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
import streamlit as st
import torch
import shap

from src.xai.shap_audit import get_model_visible_text
from src.xai.attention_audit import extract_word_attention_scores
try:
    from app.components.charts import (
        plot_horizontal_contributions,
        render_html_highlighted_text,
    )
except (ModuleNotFoundError, ImportError):
    from components.charts import (
        plot_horizontal_contributions,
        render_html_highlighted_text,
    )


def compute_shap_word_scores(
    model: torch.nn.Module,
    tokenizer: Any,
    visible_text: str,
    target_class: int,
    device: str = "cpu",
    max_length: int = 128
) -> List[Tuple[str, float]]:
    """
    Computes Word-level SHAP attribution scores using SHAP Text Masker.
    Aggregates subword attributions to words matching the visible text sequence.
    Gracefully handles edge cases and returns [] if computation fails.
    """
    if not isinstance(visible_text, str) or not visible_text.strip():
        return []

    words = visible_text.split()
    if not words:
        return []

    try:
        target_device = torch.device(device)
        model.to(target_device)
        model.eval()

        def predict_fn(texts):
            cleaned_texts = [str(t) if str(t).strip() else "" for t in texts]
            inputs = tokenizer(
                cleaned_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(target_device)
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            return probs

        masker = shap.maskers.Text(tokenizer)
        explainer = shap.Explainer(predict_fn, masker)
        
        # Compute SHAP explanation on the single sample
        explanation = explainer([visible_text])
        
        if explanation.values is None or explanation.values.ndim < 3:
            return []
        if target_class < 0 or target_class >= explanation.values.shape[2]:
            return []

        raw_tokens = explanation.data[0]
        raw_values = explanation.values[0, :, target_class]
        
        # Clean and pair tokens
        token_pairs = []
        for tok, val in zip(raw_tokens, raw_values):
            t_clean = str(tok).strip()
            if t_clean and t_clean not in ["[CLS]", "[SEP]", "[PAD]"]:
                float_val = float(val) if np.isfinite(val) else 0.0
                token_pairs.append((t_clean, float_val))
                
        # Aggregate subwords to words based on sequential matching
        word_scores: List[Tuple[str, float]] = []
        tok_idx = 0
        num_tokens = len(token_pairs)
        
        for word in words:
            # Collect tokens that form this word
            acc_str = ""
            acc_score = 0.0
            matched_any = False
            
            while tok_idx < num_tokens and len(acc_str) < len(word):
                sub_tok, sub_val = token_pairs[tok_idx]
                clean_sub = sub_tok.replace("##", "")
                acc_str += clean_sub
                acc_score += sub_val
                matched_any = True
                tok_idx += 1
                if len(acc_str) >= len(word):
                    break
                    
            if matched_any:
                word_scores.append((word, acc_score))
            else:
                word_scores.append((word, 0.0))

        # Fallback to token pairs if word reconstruction failed
        result = word_scores if word_scores else token_pairs
        return [(w, float(s)) for w, s in result if np.isfinite(s)]
    except Exception:
        return []


def compute_attention_word_scores(
    model: torch.nn.Module,
    tokenizer: Any,
    visible_text: str,
    device: str = "cpu",
    max_length: int = 128
) -> List[Tuple[str, float]]:
    """
    Extracts word-level attention scores from the final transformer layer (layer 12)
    averaged across all attention heads.
    Gracefully returns [] on empty input or extraction failure.
    """
    if not isinstance(visible_text, str) or not visible_text.strip():
        return []
    try:
        scores = extract_word_attention_scores(
            model=model,
            tokenizer=tokenizer,
            text=visible_text,
            device=device,
            max_length=max_length
        )
        return [(w, float(s)) for w, s in scores if np.isfinite(s)]
    except Exception:
        return []



def _render_token_inspector(scores: List[Tuple[str, float]], *, is_attention: bool, key: str) -> None:
    """Interactive token picker to inspect a token's local attribution value."""
    if not scores:
        return
    ranked = sorted(scores, key=lambda item: abs(item[1]), reverse=True)
    labels = [f"{word}  ·  {score:.4f}" if is_attention else f"{word}  ·  {score:+.4f}" for word, score in ranked]
    selected = st.selectbox("Inspect a token", options=list(range(len(ranked))), format_func=lambda idx: labels[idx], key=key)
    word, score = ranked[selected]
    if is_attention:
        st.caption("Selected token attention weight")
        st.metric("Attention weight", f"{score:.6f}", help="Final-layer attention weight; not a causal explanation.")
    else:
        st.caption("Selected token contribution to the predicted class")
        st.metric("SHAP attribution", f"{score:+.6f}", help="Positive values support the predicted class; negative values oppose it.")

def render_scientific_disclaimer() -> None:
    """
    Renders the mandatory scientific honesty and epistemic boundary notice.
    """
    st.warning(
        """
        **Scientific Interpretation Notice:**
        - **Attribution $\\neq$ Causal Explanation:** Feature attributions (SHAP & Attention) indicate model-behavior sensitivity under perturbation. They must **not** be interpreted as causal explanations or verified cognitive reasoning.
        - **Attention $\\neq$ Explanation Ground-Truth:** Attention did not show statistically significant behavioral relevance in the top-vs-random deletion comparison in any of the three evaluated language settings. This does not establish that attention can never be useful as an explanation.
        - **Fixed Audit Context:** All aggregate benchmark statistics displayed in this app originate from the authoritative Google Colab evaluation and are not re-estimated on individual demo inputs.
        """
    )


def render_shap_panel(
    shap_scores: List[Tuple[str, float]],
    target_class_name: str
) -> None:
    """Renders SHAP explainability analysis panel."""
    st.markdown(f"### SHAP Attribution for Target Class: `{target_class_name}`")
    st.caption(
        "Positive values (Green) indicate tokens increasing confidence in the predicted class. "
        "Negative values (Red) indicate tokens pulling confidence away toward competing classes."
    )

    if not shap_scores:
        st.info("No SHAP scores computed.")
        return

    # Highlighted inline text visualization
    st.markdown("#### Token-level view")
    html_markup = render_html_highlighted_text(shap_scores, is_attention=False)
    st.markdown(html_markup, unsafe_allow_html=True)
    _render_token_inspector(shap_scores, is_attention=False, key="shap_token_inspector")

    # Contribution Bar Chart
    st.markdown("#### Largest attribution scores")
    plot_horizontal_contributions(
        shap_scores,
        title="SHAP Attribution Breakdown",
        is_attention=False
    )


def render_attention_panel(
    attention_scores: List[Tuple[str, float]]
) -> None:
    """Renders Self-Attention explainability analysis panel."""
    st.markdown('<div class="section-eyebrow">05 / Attention signal</div>', unsafe_allow_html=True)
    st.markdown("### Final-layer self-attention")
    st.caption(
        "Extracted from mBERT's final transformer layer (Layer 12), averaged across all 12 attention heads. "
        "Represents routing weights toward the [CLS] classification token."
    )

    if not attention_scores:
        st.info("No attention scores computed.")
        return

    # Highlighted inline text visualization
    st.markdown("#### Token-level attention view")
    html_markup = render_html_highlighted_text(attention_scores, is_attention=True)
    st.markdown(html_markup, unsafe_allow_html=True)
    _render_token_inspector(attention_scores, is_attention=True, key="attention_token_inspector")

    # Contribution Bar Chart
    st.markdown("#### Highest attention weights")
    plot_horizontal_contributions(
        attention_scores,
        title="Attention Weights",
        is_attention=True
    )


def render_side_by_side_panel(
    shap_scores: List[Tuple[str, float]],
    attention_scores: List[Tuple[str, float]]
) -> None:
    """Renders direct side-by-side comparison between SHAP and Attention."""
    st.markdown('<div class="section-eyebrow">Explanation comparison</div>', unsafe_allow_html=True)
    st.markdown("### SHAP and attention side by side")
    
    st.info(
        "**Methodological Divergence Note:** SHAP and Attention represent fundamentally different attribution signals. "
        "Cross-method agreement in the research audit was weak-to-moderate (Top-1 agreement: 18%–26%; mean Spearman $\\rho$: 0.17–0.31)."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🟢 SHAP attribution")
        st.caption("Ranked by absolute attribution magnitude")
        if shap_scores:
            sorted_shap = sorted(shap_scores, key=lambda x: abs(x[1]), reverse=True)
            df_shap = pd.DataFrame([
                {
                    "Rank": i + 1,
                    "Word": w,
                    "SHAP Score": f"{s:+.4f}",
                    "Impact": "Supports" if s > 0 else "Opposes"
                }
                for i, (w, s) in enumerate(sorted_shap[:8])
            ])
            st.dataframe(df_shap, use_container_width=True, hide_index=True)
        else:
            st.write("No SHAP data.")

    with col2:
        st.markdown("#### 🔵 Attention weights")
        st.caption("Ranked by final-layer attention weight")
        if attention_scores:
            sorted_attn = sorted(attention_scores, key=lambda x: x[1], reverse=True)
            df_attn = pd.DataFrame([
                {
                    "Rank": i + 1,
                    "Word": w,
                    "Attention Weight": f"{s:.4f}",
                }
                for i, (w, s) in enumerate(sorted_attn[:8])
            ])
            st.dataframe(df_attn, use_container_width=True, hide_index=True)
        else:
            st.write("No Attention data.")

    # Top-1 Comparison observation
    if shap_scores and attention_scores:
        top_shap_word = sorted(shap_scores, key=lambda x: abs(x[1]), reverse=True)[0][0]
        top_attn_word = sorted(attention_scores, key=lambda x: x[1], reverse=True)[0][0]
        agrees = top_shap_word.lower() == top_attn_word.lower()
        
        st.markdown(
            f"""
            **Sample Agreement Observation:**
            - SHAP top word: **`{top_shap_word}`**
            - Attention top word: **`{top_attn_word}`**
            - Top-1 Match: **{'✅ MATCH' if agrees else '❌ DIVERGENCE'}**
            """
        )
