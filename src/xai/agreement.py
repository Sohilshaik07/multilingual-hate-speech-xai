"""
src/xai/agreement.py

Evaluation module for cross-method explainability agreement between Self-Attention and SHAP.

Metrics Implemented:
1. Top-Token Agreement:
   - Measures exact top-1 word agreement rate between self-attention and SHAP.
2. Word Attribution Rank Correlation:
   - Evaluates Spearman rank correlation coefficient (rho) over word attributions per example.
3. Distribution Statistics:
   - Mean, median, and standard deviation of Spearman rho.
   - Rates of positive correlation (rho > 0), negative correlation (rho < 0), and near-zero correlation (|rho| < 0.1).
"""

from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def compute_top_token_agreement(
    attention_top_words: List[str],
    shap_top_words: List[str]
) -> Dict[str, Any]:
    """
    Computes exact top-1 word agreement between attention and SHAP.
    
    Returns:
        Dictionary with agreement rate and matching count.
    """
    if len(attention_top_words) != len(shap_top_words):
        raise ValueError("Lengths of attention_top_words and shap_top_words must match.")
    
    n_examples = len(attention_top_words)
    if n_examples == 0:
        return {"n_examples": 0, "agreement_rate": 0.0, "agreement_count": 0}
        
    matches = [
        str(a).strip().lower() == str(s).strip().lower()
        for a, s in zip(attention_top_words, shap_top_words)
    ]
    count = int(sum(matches))
    rate = float(count / n_examples)
    
    return {
        "n_examples": n_examples,
        "agreement_count": count,
        "agreement_rate": rate,
        "matches": matches
    }


def compute_sample_spearman(
    attention_scores: List[float],
    shap_scores: List[float]
) -> Tuple[float, float]:
    """
    Computes Spearman rank correlation between attention scores and SHAP scores
    for a single example.
    
    Returns:
        (rho, p_value)
    """
    if len(attention_scores) < 3 or len(shap_scores) < 3:
        return 0.0, 1.0
        
    # Check for constant arrays
    if np.all(np.array(attention_scores) == attention_scores[0]) or np.all(np.array(shap_scores) == shap_scores[0]):
        return 0.0, 1.0
        
    res = spearmanr(attention_scores, shap_scores)
    rho = float(res.statistic) if not np.isnan(res.statistic) else 0.0
    pval = float(res.pvalue) if not np.isnan(res.pvalue) else 1.0
    return rho, pval


def compute_agreement_summary(agreement_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes complete agreement and correlation distribution statistics from
    an agreement DataFrame (such as hecm_attention_shap_agreement.csv).
    
    Expected columns:
    - 'top_attention_word' / 'attention_top'
    - 'top_shap_word' / 'shap_top'
    - 'spearman_rho' / 'spearman_r'
    """
    df = agreement_df.copy()
    
    # Identify attention/shap columns
    attn_col = "top_attention_word" if "top_attention_word" in df.columns else "attention_top"
    shap_col = "top_shap_word" if "top_shap_word" in df.columns else "shap_top"
    rho_col = "spearman_rho" if "spearman_rho" in df.columns else "spearman_r"
    
    n_examples = len(df)
    
    # Top word agreement
    agree_metrics = compute_top_token_agreement(df[attn_col].tolist(), df[shap_col].tolist())
    
    # Spearman distribution
    if rho_col in df.columns:
        rhos = df[rho_col].dropna().values
        mean_rho = float(np.mean(rhos))
        median_rho = float(np.median(rhos))
        std_rho = float(np.std(rhos))
        pos_rate = float(np.mean(rhos > 0))
        neg_rate = float(np.mean(rhos < 0))
        near_zero_rate = float(np.mean(np.abs(rhos) < 0.1))
    else:
        mean_rho, median_rho, std_rho = np.nan, np.nan, np.nan
        pos_rate, neg_rate, near_zero_rate = np.nan, np.nan, np.nan
        
    return {
        "n_examples": n_examples,
        "top_word_agreement": agree_metrics["agreement_rate"],
        "top_word_agreement_count": agree_metrics["agreement_count"],
        "mean_spearman": mean_rho,
        "median_spearman": median_rho,
        "std_spearman": std_rho,
        "positive_rho_rate": pos_rate,
        "negative_rho_rate": neg_rate,
        "near_zero_rho_rate": near_zero_rate
    }
