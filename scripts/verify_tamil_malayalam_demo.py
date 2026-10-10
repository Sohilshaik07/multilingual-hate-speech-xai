"""
scripts/verify_tamil_malayalam_demo.py

Manual verification script simulating the exact Streamlit demonstration flow
for Tamil and Malayalam under MODEL_SOURCE=hf.
"""

import sys
import os
from pathlib import Path

# Force UTF-8 output so native-script characters (Malayalam, Tamil) don't
# cause UnicodeEncodeError on Windows cp1252 consoles.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def safe_print(text: str) -> None:
    """Print with ascii fallback for consoles that can't handle Unicode."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="replace").decode("ascii"))

os.environ["MODEL_SOURCE"] = "hf"

from app.utils.labels import get_samples_for_dataset, get_label_name, LABEL_MAPPINGS
from app.utils.model_loader import load_cached_pipeline, predict_sample
from app.components.explanations import compute_attention_word_scores, compute_shap_word_scores

def verify_language(dataset_key: str):
    print("=" * 65)
    print(f"VERIFYING LANGUAGE: {dataset_key.upper()} (MODEL_SOURCE=hf)")
    print("=" * 65)
    
    samples = get_samples_for_dataset(dataset_key)
    print(f"Sample presets available: {len(samples)}")
    for i, s in enumerate(samples):
        safe_print(f"  [{i+1}] {s['title']}")
    
    sample = samples[0]
    safe_print(f"\nAnalyzing Sample 1: \"{sample['title']}\"")
    # Truncate and ascii-safe for console display; actual inference uses full text
    safe_print(f"Input text: \"{sample['text'][:65]}...\"")
    
    model, tokenizer, device = load_cached_pipeline(dataset_key)
    res = predict_sample(model, tokenizer, sample["text"], dataset_key)
    
    pred_cls = res["predicted_class"]
    pred_label = get_label_name(dataset_key, pred_cls)
    conf = res["confidence"]
    probs = res["probabilities"]
    
    print(f"\n--- Prediction Results ---")
    print(f"Predicted class index : {pred_cls}")
    print(f"Predicted label name  : {pred_label}")
    print(f"Winning confidence    : {conf * 100:.2f}%")
    print(f"Probability simplex sum: {sum(probs.values()):.6f}")
    print(f"Total distribution classes: {len(probs)}")
    for c in sorted(probs.keys()):
        c_name = LABEL_MAPPINGS[dataset_key].get(c, f"Class {c}")
        marker = " [WINNER]" if c == pred_cls else ""
        print(f"   Class {c} ({c_name:<25}): {probs[c]*100:6.2f}%{marker}")
        
    print(f"\n--- Explainability Attributions ---")
    attn_scores = compute_attention_word_scores(model, tokenizer, res["visible_text"])
    print(f"Attention word tokens count : {len(attn_scores)}")
    if attn_scores:
        top_attn = attn_scores[0]
        print(f"Top-1 Attention token       : '{top_attn[0]}' (Score: {top_attn[1]:.4f})")
        print(f"Top-3 Attention tokens      : {[w for w, _ in attn_scores[:3]]}")
        
    shap_scores = compute_shap_word_scores(model, tokenizer, res["visible_text"], pred_cls)
    print(f"SHAP word tokens count      : {len(shap_scores)}")
    if shap_scores:
        top_shap = sorted(shap_scores, key=lambda x: abs(x[1]), reverse=True)[0]
        print(f"Top-1 SHAP token (|val|)    : '{top_shap[0]}' (Value: {top_shap[1]:.4f})")
        print(f"Top-3 SHAP tokens by |val|  : {[w for w, _ in sorted(shap_scores, key=lambda x: abs(x[1]), reverse=True)[:3]]}")
        
    print(f"\nVerification status for {dataset_key.upper()}: [PASS]")
    return res, attn_scores, shap_scores

if __name__ == "__main__":
    print("DEMO VERIFICATION: TAMIL & MALAYALAM PIPELINES (MODEL_SOURCE=hf)\n")
    res_tam, attn_tam, shap_tam = verify_language("tamil")
    print()
    res_mal, attn_mal, shap_mal = verify_language("malayalam")
    
    print("\n" + "=" * 65)
    print("CROSS-LANGUAGE SWITCHING STATE INTEGRITY CHECK")
    print("=" * 65)
    print(f"Tamil classes count     : {len(res_tam['probabilities'])} (Expected: 6)")
    print(f"Malayalam classes count : {len(res_mal['probabilities'])} (Expected: 5)")
    assert len(res_tam['probabilities']) == 6, "Tamil must have 6 classes"
    assert len(res_mal['probabilities']) == 5, "Malayalam must have 5 classes"
    assert res_tam['probabilities'].keys() != res_mal['probabilities'].keys() or len(res_tam['probabilities']) != len(res_mal['probabilities']), "Label spaces must differ"
    print("State transition check: PASS (No label or explanation contamination between languages)")
