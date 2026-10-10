"""Presentation components for the validated mBERT prediction output."""
from html import escape
from typing import Dict, Any
import pandas as pd
import streamlit as st

try:
    from app.components.charts import plot_probability_distribution
    from app.utils.labels import get_label_name
except (ModuleNotFoundError, ImportError):
    from components.charts import plot_probability_distribution
    from utils.labels import get_label_name


def render_prediction_panel(prediction_result: Dict[str, Any], dataset_key: str) -> None:
    """Render prediction summary, probabilities, and token-limit context."""
    pred_cls = prediction_result["predicted_class"]
    conf = float(prediction_result["confidence"])
    probs = prediction_result["probabilities"]
    label_map = prediction_result["label_mapping"]
    visible_text = prediction_result["visible_text"]
    token_count = prediction_result["token_count"]
    label_str = get_label_name(dataset_key, pred_cls)
    dark = bool(st.session_state.get("dark_mode", True))
    panel_bg = "#132438" if dark else "#e9f7f5"
    panel_text = "#ffffff" if dark else "#17364a"
    panel_muted = "#a9bdcc" if dark else "#607789"
    field_bg = "#0e1929" if dark else "#fbfcfe"
    field_border = "#29394d" if dark else "#e4eaf0"
    field_text = "#e7eef8" if dark else "#253449"

    st.markdown('<div class="section-eyebrow">03 / Model output</div>', unsafe_allow_html=True)
    st.markdown("### Classification result")
    st.caption("This is the model's predicted category, not a ground-truth label or a definitive judgment about the text.")

    # A single, compact result banner makes the prediction the visual anchor.
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;justify-content:space-between;gap:1rem;flex-wrap:wrap;
                    padding:1.1rem 1.25rem;background:{panel_bg};border-radius:16px;margin:.5rem 0 1rem;">
          <div>
            <div style="color:{panel_muted};font-size:.7rem;font-weight:800;letter-spacing:.13em;text-transform:uppercase;margin-bottom:.35rem;">Predicted category</div>
            <div style="font-family:Manrope,sans-serif;color:{panel_text};font-size:1.65rem;font-weight:800;letter-spacing:-.04em;">{escape(str(label_str))}</div>
          </div>
          <div style="min-width:145px;padding:.65rem .9rem;border:1px solid rgba(255,255,255,.16);border-radius:12px;background:rgba(255,255,255,.06);">
            <div style="color:{panel_muted};font-size:.7rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;">Raw confidence</div>
            <div style="color:#2dd4bf;font-size:1.5rem;font-weight:800;">{conf * 100:.2f}%</div>
          </div>
        </div>
        """, unsafe_allow_html=True,
    )

    c1, c2 = st.columns([1.35, 1])
    with c1:
        st.markdown("**Text seen by the model**")
        st.markdown(
            f'<div style="padding:.85rem 1rem;border:1px solid {field_border};background:{field_bg};border-radius:12px;line-height:1.7;color:{field_text};">{escape(str(visible_text))}</div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown("**Input processing**")
        st.metric("Tokens processed", f"{token_count} / 128", help="Includes special tokens; the model content limit is 126 tokens.")

    if token_count >= 126:
        st.warning("Sequence boundary: the input approaches or exceeds the 126 content-token limit. Explanations apply only to the text ingested by the transformer.")

    st.markdown("#### Class probability distribution")
    st.caption("The winning class is highlighted. Probabilities are raw softmax outputs and are not post-hoc calibrated.")
    plot_probability_distribution(probs, label_map, pred_cls)

    with st.expander("Inspect exact probability values"):
        df = pd.DataFrame([
            {
                "Class index": c,
                "Category": label_map.get(c, f"Class {c}"),
                "Raw probability": f"{probs[c]:.6f}",
                "Percentage": f"{probs[c] * 100:.2f}%",
                "Predicted": "✓" if c == pred_cls else "",
            }
            for c in sorted(probs.keys())
        ])
        st.dataframe(df, use_container_width=True, hide_index=True)
