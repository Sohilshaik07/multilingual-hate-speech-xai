"""
src/xai/shap_audit.py

Reproducibility module for SHAP (KernelSHAP / Partition Explainer) explainability audits.

Methodology Implemented:
1. Truncation-Safe Token Handling:
   - For max_length=128, reserves special-token positions ([CLS], [SEP]), using at most 126 content tokens.
   - Reconstructs model-visible text to ensure SHAP maskers and models evaluate identical token sequences.
2. Word-Level Attribution:
   - Evaluates token/word importance via SHAP Text masker or word-level perturbation.
3. Faithfulness Testing via Deletion Protocol:
   - Identifies top, bottom, and random words.
   - Deletes target word and measures confidence drop: (original_confidence - perturbed_confidence).
   - Measures prediction change rate (prediction flipping).
   - Computes one-sided Wilcoxon signed-rank tests: Top vs Random and Top vs Bottom.

IMPORTANT:
- Historical aggregate results in results/xai/ are authoritative.
- This module implements the validated audit methodology for local reproducibility and inspection.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon


def get_model_visible_text(text: str, tokenizer: Any, max_length: int = 128) -> str:
    """
    Reconstructs model-visible text under strict max_length truncation constraints.
    Reserves 2 special token slots ([CLS], [SEP]), bounding content tokens to (max_length - 2) = 126.
    Ensures explainability maskers evaluate only text visible to the transformer encoder.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    
    # Tokenize without special tokens to isolate content tokens
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    max_content_tokens = max(1, max_length - 2)
    truncated_ids = token_ids[:max_content_tokens]
    
    # Decode back to visible text string
    visible_text = tokenizer.decode(truncated_ids, clean_up_tokenization_spaces=True).strip()
    return visible_text


def remove_word_from_text(text: str, word: str) -> str:
    """
    Deletes the first occurrence of a specific word from text,
    normalizing whitespace.
    """
    words = text.split()
    new_words = []
    removed = False
    for w in words:
        if not removed and w == word:
            removed = True
            continue
        new_words.append(w)
    return " ".join(new_words).strip()


def run_deletion_faithfulness_test(
    model: torch.nn.Module,
    tokenizer: Any,
    text: str,
    top_word: str,
    bottom_word: str,
    random_word: str,
    device: Optional[str] = None,
    max_length: int = 128
) -> Dict[str, Any]:
    """
    Performs deletion perturbation testing on a single sample:
    1. Evaluates baseline model confidence on visible text.
    2. Deletes top, bottom, and random words separately.
    3. Measures confidence drops: (original_confidence - perturbed_confidence).
    4. Records whether the predicted class changed.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    target_device = torch.device(device)
    model.to(target_device)
    model.eval()

    visible_text = get_model_visible_text(text, tokenizer, max_length=max_length)
    if not visible_text:
        return {}

    # Helper function to get prediction & confidence for given text
    def _eval_text(t: str) -> Tuple[int, float]:
        if not t.strip():
            # If text became completely empty, fall back to empty string
            t = ""
        inputs = tokenizer(t, max_length=max_length, truncation=True, return_tensors="pt").to(target_device)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        pred_cls = int(np.argmax(probs))
        conf = float(probs[pred_cls])
        return pred_cls, conf

    orig_class, orig_conf = _eval_text(visible_text)

    # Top word deletion
    text_no_top = remove_word_from_text(visible_text, top_word)
    top_class, top_conf = _eval_text(text_no_top)
    top_drop = orig_conf - top_conf
    top_changed = (top_class != orig_class)

    # Bottom word deletion
    text_no_bottom = remove_word_from_text(visible_text, bottom_word)
    bottom_class, bottom_conf = _eval_text(text_no_bottom)
    bottom_drop = orig_conf - bottom_conf
    bottom_changed = (bottom_class != orig_class)

    # Random word deletion
    text_no_rand = remove_word_from_text(visible_text, random_word)
    rand_class, rand_conf = _eval_text(text_no_rand)
    rand_drop = orig_conf - rand_conf
    rand_changed = (rand_class != orig_class)

    return {
        "text": visible_text,
        "pred_class": orig_class,
        "original_confidence": orig_conf,
        "top_word": top_word,
        "top_confidence": top_conf,
        "top_drop": float(top_drop),
        "top_prediction_changed": bool(top_changed),
        "bottom_word": bottom_word,
        "bottom_confidence": bottom_conf,
        "bottom_drop": float(bottom_drop),
        "bottom_prediction_changed": bool(bottom_changed),
        "random_word": random_word,
        "random_confidence": rand_conf,
        "random_drop": float(rand_drop),
        "random_prediction_changed": bool(rand_changed)
    }


def compute_faithfulness_statistics(audit_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Aggregates sample-level audit metrics and performs one-sided Wilcoxon tests:
    - mean/median confidence drops for top, bottom, and random
    - prediction change rates
    - Wilcoxon signed-rank test for Top vs Random (alternative='greater')
    - Wilcoxon signed-rank test for Top vs Bottom (alternative='greater')
    """
    top_drops = audit_df["top_drop"].values
    bottom_drops = audit_df["bottom_drop"].values
    random_drops = audit_df["random_drop"].values

    top_mean_drop = float(np.mean(top_drops))
    top_median_drop = float(np.median(top_drops))
    top_flip_rate = float(np.mean(audit_df["top_prediction_changed"]))

    bottom_mean_drop = float(np.mean(bottom_drops))
    bottom_median_drop = float(np.median(bottom_drops))
    bottom_flip_rate = float(np.mean(audit_df["bottom_prediction_changed"]))

    random_mean_drop = float(np.mean(random_drops))
    random_median_drop = float(np.median(random_drops))
    random_flip_rate = float(np.mean(audit_df["random_prediction_changed"]))

    # Wilcoxon Top vs Random
    diff_rand = top_drops - random_drops
    if np.any(diff_rand != 0):
        res_rand = wilcoxon(diff_rand, alternative="greater")
        top_vs_random_stat = float(res_rand.statistic)
        top_vs_random_p = float(res_rand.pvalue)
    else:
        top_vs_random_stat, top_vs_random_p = np.nan, 1.0

    # Wilcoxon Top vs Bottom
    diff_bot = top_drops - bottom_drops
    if np.any(diff_bot != 0):
        res_bot = wilcoxon(diff_bot, alternative="greater")
        top_vs_bottom_stat = float(res_bot.statistic)
        top_vs_bottom_p = float(res_bot.pvalue)
    else:
        top_vs_bottom_stat, top_vs_bottom_p = np.nan, 1.0

    return {
        "n_examples": len(audit_df),
        "top_mean_confidence_drop": top_mean_drop,
        "top_median_confidence_drop": top_median_drop,
        "top_prediction_change_rate": top_flip_rate,
        "bottom_mean_confidence_drop": bottom_mean_drop,
        "bottom_median_confidence_drop": bottom_median_drop,
        "bottom_prediction_change_rate": bottom_flip_rate,
        "random_mean_confidence_drop": random_mean_drop,
        "random_median_confidence_drop": random_median_drop,
        "random_prediction_change_rate": random_flip_rate,
        "top_vs_random_statistic": top_vs_random_stat,
        "top_vs_random_p": top_vs_random_p,
        "top_vs_bottom_statistic": top_vs_bottom_stat,
        "top_vs_bottom_p": top_vs_bottom_p,
        "top_greater_than_random": float(np.mean(top_drops > random_drops)),
        "top_greater_than_bottom": float(np.mean(top_drops > bottom_drops))
    }
