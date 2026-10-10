"""Components package."""

from .prediction import render_prediction_panel
from .explanations import (
    compute_shap_word_scores,
    compute_attention_word_scores,
    render_shap_panel,
    render_attention_panel,
    render_side_by_side_panel,
    render_scientific_disclaimer,
)
from .charts import (
    plot_probability_distribution,
    plot_horizontal_contributions,
    render_html_highlighted_text,
)

__all__ = [
    "render_prediction_panel",
    "compute_shap_word_scores",
    "compute_attention_word_scores",
    "render_shap_panel",
    "render_attention_panel",
    "render_side_by_side_panel",
    "render_scientific_disclaimer",
    "plot_probability_distribution",
    "plot_horizontal_contributions",
    "render_html_highlighted_text",
]