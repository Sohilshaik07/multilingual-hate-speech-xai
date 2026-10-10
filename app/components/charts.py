"""Clean, consistent visualizations for probabilities and token attributions."""
from html import escape
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

TEAL = "#2dd4bf"
BLUE = "#7c9cff"
GREEN = "#34d399"
RED = "#fb7185"

def _palette():
    dark = bool(st.session_state.get("dark_mode", True))
    return {
        "ink": "#e7eef8" if dark else "#243247",
        "muted": "#9aacc1" if dark else "#718096",
        "grid": "#29394d" if dark else "#e7ecf2",
        "canvas": "#111d2e" if dark else "#ffffff",
        "neutral": "#34465d" if dark else "#cbd5e1",
    }

def _polish_axis(ax) -> None:
    palette = _palette()
    ax.set_facecolor(palette["canvas"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(palette["grid"])
    ax.tick_params(axis="y", length=0, colors=palette["ink"], labelsize=9)
    ax.tick_params(axis="x", colors=palette["muted"], labelsize=8, length=0, pad=7)
    ax.grid(axis="x", linestyle="-", linewidth=.7, alpha=1, color=palette["grid"], zorder=0)
    ax.set_axisbelow(True)


def plot_probability_distribution(probabilities: Dict[int, float], label_mapping: Dict[int, str], predicted_class: int) -> None:
    if not probabilities:
        st.info("No class probabilities are available.")
        return
    classes = sorted(probabilities.keys(), key=lambda c: probabilities[c], reverse=True)
    values = [float(probabilities[c]) for c in classes]
    labels = [str(label_mapping.get(c, f"Class {c}")) for c in classes]
    palette = _palette()
    fig, ax = plt.subplots(figsize=(8.4, max(2.15, len(classes) * .48)), dpi=140)
    fig.patch.set_facecolor(palette["canvas"])
    colors = [TEAL if c == predicted_class else palette["neutral"] for c in classes]
    y = np.arange(len(classes))
    bars = ax.barh(y, values, color=colors, height=.54, edgecolor="none", zorder=3)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.14)
    ax.set_xlabel("Raw softmax probability", fontsize=8.5, color=palette["muted"], labelpad=10)
    _polish_axis(ax)
    for bar, value, cls in zip(bars, values, classes):
        ax.text(min(value + .018, 1.025), bar.get_y() + bar.get_height()/2, f"{value*100:.2f}%",
                va="center", ha="left", fontsize=8.5, color=TEAL if cls == predicted_class else palette["ink"],
                fontweight="bold" if cls == predicted_class else "medium")
    fig.tight_layout(pad=1.15)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def plot_horizontal_contributions(contributions: List[Tuple[str, float]], title: str, is_attention: bool = False, top_k: int = 12) -> None:
    if not contributions:
        st.info("No token attributions available to display.")
        return
    ranked = sorted(contributions, key=lambda item: abs(item[1]), reverse=True)[:top_k]
    ranked.reverse()
    words = [str(item[0]) for item in ranked]
    scores = [float(item[1]) for item in ranked]
    palette = _palette()
    fig, ax = plt.subplots(figsize=(8.4, max(2.5, len(words) * .39)), dpi=140)
    fig.patch.set_facecolor(palette["canvas"])
    colors = [BLUE] * len(scores) if is_attention else [GREEN if score >= 0 else RED for score in scores]
    y = np.arange(len(words))
    bars = ax.barh(y, scores, color=colors, height=.56, edgecolor="none", zorder=3)
    ax.set_yticks(y, words)
    _polish_axis(ax)
    minimum, maximum = min(scores), max(scores)
    margin = max(abs(minimum), abs(maximum), .0001) * .34
    if is_attention:
        ax.set_xlim(0, max(maximum + margin, .001))
        ax.set_xlabel("Final-layer attention weight", fontsize=8.5, color=palette["muted"], labelpad=10)
    else:
        ax.set_xlim(min(0, minimum - margin), max(0, maximum + margin))
        ax.axvline(0, color="#a9b5c4", linewidth=.8, zorder=2)
        ax.set_xlabel("Attribution score · positive supports / negative opposes predicted class", fontsize=8.5, color=palette["muted"], labelpad=10)
    for bar, score in zip(bars, scores):
        text = f"{score:.4f}" if is_attention else f"{score:+.4f}"
        offset = margin * .09 if score >= 0 else -margin * .09
        ax.text(score + offset, bar.get_y() + bar.get_height()/2, text, va="center",
                ha="left" if score >= 0 else "right", fontsize=8, color=palette["ink"], fontweight="medium")
    fig.tight_layout(pad=1.15)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_html_highlighted_text(contributions: List[Tuple[str, float]], is_attention: bool = False) -> str:
    """Build an escaped HTML token ribbon; input tokens are untrusted user text."""
    if not contributions:
        return ""
    max_mag = max([abs(float(score)) for _, score in contributions] + [1e-8])
    dark = bool(st.session_state.get("dark_mode", True))
    ribbon_bg = "#0e1929" if dark else "#fbfcfe"
    ribbon_border = "#29394d" if dark else "#e4eaf0"
    spans = []
    for word, score in contributions:
        score = float(score)
        intensity = min(.8, max(.12, abs(score) / max_mag * .72))
        alpha = min(.78, max(.20, intensity * 1.45)) if dark else intensity
        if is_attention:
            background, foreground = f"rgba(69,101,207,{alpha:.2f})", ("#eff4ff" if dark else "#253449")
        elif score >= 0:
            background, foreground = f"rgba(22,163,119,{alpha:.2f})", ("#eafff6" if dark else "#164e42")
        else:
            background, foreground = f"rgba(214,90,98,{alpha:.2f})", ("#fff0f1" if dark else "#742f36")
        label = escape(str(word))
        tooltip = escape(f"{word}: {score:.4f}" if is_attention else f"{word}: {score:+.4f}", quote=True)
        spans.append(
            f'<span title="{tooltip}" style="background:{background};color:{foreground};padding:5px 8px;'
            f'margin:3px 3px 3px 0;border:1px solid rgba(100,116,139,.10);border-radius:7px;'
            f'display:inline-block;font-size:.9rem;line-height:1.5;font-weight:600;">{label}</span>'
        )
    return (f'<div style="background:{ribbon_bg};border:1px solid {ribbon_border};border-radius:12px;'
            'padding:14px 16px;line-height:2.5;margin:.35rem 0 1rem;">' + " ".join(spans) + '</div>')
