"""
src/xai/attention_audit.py

Reproducibility module for self-attention explainability audits.

Methodology Implemented:
1. Attention Extraction:
   - Evaluates encoder with output_attentions=True.
   - Extracts attention weights from the final transformer layer (layer 12 for mBERT).
   - Averages attention across all 12 attention heads.
2. Word-Level Importance Aggregation:
   - Aggregates subword attention scores to word-level tokens using tokenizer word offsets.
   - Disregards special tokens ([CLS], [SEP], [PAD]).
3. Faithfulness Testing via Deletion:
   - Identifies top, bottom, and randomly chosen words.
   - Measures confidence drop upon word deletion.
   - Computes Wilcoxon signed-rank tests: Top vs Random and Top vs Bottom.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon

from src.xai.shap_audit import get_model_visible_text, remove_word_from_text


def extract_word_attention_scores(
    model: torch.nn.Module,
    tokenizer: Any,
    text: str,
    device: Optional[str] = None,
    max_length: int = 128
) -> List[Tuple[str, float]]:
    """
    Extracts word-level attention attribution scores from the final transformer layer.
    
    Returns:
        List of (word, attention_score) tuples ordered by appearance.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    target_device = torch.device(device)
    model.to(target_device)
    model.eval()

    visible_text = get_model_visible_text(text, tokenizer, max_length=max_length)
    if not visible_text:
        return []

    words = visible_text.split()
    if not words:
        return []

    # Encode with offset mapping to link subwords to word spans
    encoding = tokenizer(
        visible_text,
        max_length=max_length,
        truncation=True,
        return_offsets_mapping=True,
        return_tensors="pt"
    )
    
    input_ids = encoding["input_ids"].to(target_device)
    attention_mask = encoding["attention_mask"].to(target_device)
    offset_mapping = encoding["offset_mapping"][0].cpu().numpy()

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True
        )

    # Final layer attentions: shape (1, num_heads, seq_len, seq_len)
    last_layer_attn = outputs.attentions[-1]
    # Average across all attention heads: shape (seq_len, seq_len)
    attn_matrix = last_layer_attn[0].mean(dim=0).cpu().numpy()

    # CLS attention to all tokens (index 0): shape (seq_len,)
    cls_attn = attn_matrix[0]

    # Map subwords to words based on character span alignment
    word_scores = []
    current_char = 0
    
    for word in words:
        start_char = visible_text.find(word, current_char)
        if start_char == -1:
            start_char = current_char
        end_char = start_char + len(word)
        current_char = end_char

        # Find all subwords covering this character range
        subword_indices = []
        for idx, (sub_start, sub_end) in enumerate(offset_mapping):
            if sub_start == 0 and sub_end == 0:
                continue  # Special token
            if max(start_char, sub_start) < min(end_char, sub_end):
                subword_indices.append(idx)

        if subword_indices:
            score = float(np.mean(cls_attn[subword_indices]))
        else:
            score = 0.0
            
        word_scores.append((word, score))

    return word_scores


def run_attention_sample_audit(
    model: torch.nn.Module,
    tokenizer: Any,
    text: str,
    device: Optional[str] = None,
    max_length: int = 128,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Performs full attention deletion audit for a single input text.
    """
    visible_text = get_model_visible_text(text, tokenizer, max_length=max_length)
    if not visible_text:
        return {}

    word_scores = extract_word_attention_scores(
        model=model,
        tokenizer=tokenizer,
        text=visible_text,
        device=device,
        max_length=max_length
    )
    if len(word_scores) < 2:
        return {}

    # Identify top, bottom, and random words
    sorted_words = sorted(word_scores, key=lambda x: x[1], reverse=True)
    top_word = sorted_words[0][0]
    bottom_word = sorted_words[-1][0]

    rng = np.random.RandomState(seed + len(visible_text))
    rand_idx = rng.randint(0, len(word_scores))
    random_word = word_scores[rand_idx][0]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    target_device = torch.device(device)

    def _eval_text(t: str) -> Tuple[int, float]:
        if not t.strip():
            t = ""
        inputs = tokenizer(t, max_length=max_length, truncation=True, return_tensors="pt").to(target_device)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        cls_idx = int(np.argmax(probs))
        return cls_idx, float(probs[cls_idx])

    orig_cls, orig_conf = _eval_text(visible_text)

    # Top deletion
    t_top = remove_word_from_text(visible_text, top_word)
    c_top, conf_top = _eval_text(t_top)
    top_drop = orig_conf - conf_top

    # Bottom deletion
    t_bot = remove_word_from_text(visible_text, bottom_word)
    c_bot, conf_bot = _eval_text(t_bot)
    bot_drop = orig_conf - conf_bot

    # Random deletion
    t_rand = remove_word_from_text(visible_text, random_word)
    c_rand, conf_rand = _eval_text(t_rand)
    rand_drop = orig_conf - conf_rand

    return {
        "text": visible_text,
        "pred_class": orig_cls,
        "original_confidence": orig_conf,
        "top_word": top_word,
        "top_confidence": conf_top,
        "top_drop": float(top_drop),
        "top_prediction_changed": bool(c_top != orig_cls),
        "bottom_word": bottom_word,
        "bottom_confidence": conf_bot,
        "bottom_drop": float(bot_drop),
        "bottom_prediction_changed": bool(c_bot != orig_cls),
        "random_word": random_word,
        "random_confidence": conf_rand,
        "random_drop": float(rand_drop),
        "random_prediction_changed": bool(c_rand != orig_cls)
    }
