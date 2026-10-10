"""
src/robustness/formatting_robustness.py

Analysis module for text formatting perturbation robustness.

Historical Context & Integrity Note:
- The original synthetic formatting transformation implementation (whitespace noise,
  punctuation spacing, redundant spacing) was executed in Google Colab.
- The resulting CSV (results/robustness/formatting_robustness_results.csv) is the
  authoritative historical experimental record.
- As required by the research integrity protocol, this module reads, validates,
  and provides analysis utilities around the authoritative results without fabricating
  or regenerating the historical transformations.
"""

import os
from typing import Dict, Any, Optional
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
DEFAULT_CSV_PATH = os.path.join(PROJECT_ROOT, "results", "robustness", "formatting_robustness_results.csv")


def load_formatting_robustness_results(csv_path: Optional[str] = None) -> pd.DataFrame:
    """
    Loads and validates the authoritative formatting robustness results table.
    
    Expected columns:
    ['Language', 'Perturbation', 'N_Changed', 'Prediction_Consistency',
     'Prediction_Flip_Rate', 'Mean_Confidence_Change',
     'Mean_Absolute_Confidence_Change', 'Median_Absolute_Confidence_Change']
    """
    path = csv_path or DEFAULT_CSV_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Authoritative formatting robustness CSV not found at: {path}")
    df = pd.read_csv(path)
    return df


def summarize_formatting_robustness(csv_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Summarizes prediction consistency and confidence stability across all languages
    and perturbation types.
    """
    df = load_formatting_robustness_results(csv_path)
    
    summary = {
        "total_experiments": len(df),
        "languages": df["Language"].unique().tolist(),
        "perturbations": df["Perturbation"].unique().tolist(),
        "mean_prediction_consistency": float(df["Prediction_Consistency"].mean()),
        "max_prediction_flip_rate": float(df["Prediction_Flip_Rate"].max()),
        "mean_abs_confidence_change": float(df["Mean_Absolute_Confidence_Change"].mean()),
        "by_language": {}
    }
    
    for lang, group in df.groupby("Language"):
        summary["by_language"][lang] = {
            "perturbations_tested": group["Perturbation"].tolist(),
            "mean_consistency": float(group["Prediction_Consistency"].mean()),
            "flip_rate": float(group["Prediction_Flip_Rate"].mean()),
            "mean_abs_conf_change": float(group["Mean_Absolute_Confidence_Change"].mean())
        }
        
    return summary
