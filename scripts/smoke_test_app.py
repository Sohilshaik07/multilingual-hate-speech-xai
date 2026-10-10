"""
scripts/smoke_test_app.py

Smoke test for the Phase 7 Streamlit demonstration application.
Validates loading and end-to-end inference + explainability for all 3 languages.
"""

import sys
from pathlib import Path
import os
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Ensure UTF-8 output on Windows
sys.stdout.reconfigure(encoding="utf-8")

# Set project root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.utils.labels import get_samples_for_dataset, get_label_name
from app.utils.model_loader import (
    load_cached_pipeline, predict_sample, get_checkpoint_storage_info,
    get_model_source,
)
from app.components.explanations import compute_attention_word_scores, compute_shap_word_scores

def run_smoke_test():
    print("=" * 60)
    print("PHASE 7 STREAMLIT APPLICATION SMOKE TEST")
    print("=" * 60)

    model_source = get_model_source()
    print(f"Model source: {model_source.upper()}")

    # 1. Storage check — only meaningful for local source
    storage = get_checkpoint_storage_info()
    print(f"Total Checkpoint Size (local): {storage['total_size_gb']} GB ({storage['total_size_mb']} MB)")
    if model_source == "local":
        assert storage['total_size_gb'] > 1.8, "Storage size abnormal! Expected > 1.8 GB for 3 local checkpoints."
        print("  [PASS] Local storage size verified (> 1.8 GB).")
    else:
        print("  [SKIP] Storage size assertion skipped (MODEL_SOURCE=hf; weights streamed from Hub).")

    # 2. Test each language
    languages = [
        ("hecm", "Hindi-English"),
        ("tamil", "Tamil-English"),
        ("malayalam", "Malayalam-English")
    ]

    for lang_key, lang_name in languages:
        print(f"\n--- Testing {lang_name} ({lang_key.upper()}) ---")
        
        # Load model and tokenizer
        model, tokenizer, device = load_cached_pipeline(lang_key, device="cpu")
        print(f"  [PASS] Checkpoint loaded on {device}")
        
        # Get sample text
        samples = get_samples_for_dataset(lang_key)
        sample_text = samples[0]["text"]
        print(f"  Input: \"{sample_text[:50]}...\"")
        
        # Inference
        pred_res = predict_sample(model, tokenizer, sample_text, lang_key, device="cpu")
        pred_cls = pred_res["predicted_class"]
        conf = pred_res["confidence"]
        label_str = get_label_name(lang_key, pred_cls)
        print(f"  [PASS] Prediction: Class {pred_cls} ({label_str}) with confidence {conf * 100:.2f}%")
        assert 0.0 <= conf <= 1.0, "Confidence out of bounds!"
        assert abs(sum(pred_res["probabilities"].values()) - 1.0) < 1e-4, "Probabilities do not sum to 1!"
        
        # Attention
        attn_scores = compute_attention_word_scores(
            model=model,
            tokenizer=tokenizer,
            visible_text=pred_res["visible_text"],
            device="cpu"
        )
        print(f"  [PASS] Attention: {len(attn_scores)} word attributions extracted. Top: {attn_scores[0] if attn_scores else 'None'}")
        assert len(attn_scores) > 0, "Attention scores empty!"
        
        # SHAP
        shap_scores = compute_shap_word_scores(
            model=model,
            tokenizer=tokenizer,
            visible_text=pred_res["visible_text"],
            target_class=pred_cls,
            device="cpu"
        )
        print(f"  [PASS] SHAP: {len(shap_scores)} word attributions extracted. Top: {shap_scores[0] if shap_scores else 'None'}")
        assert len(shap_scores) > 0, "SHAP scores empty!"

    print("\n" + "=" * 60)
    print("ALL 3 LANGUAGE PIPELINES VERIFIED SUCCESSFULLY: PASS")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = run_smoke_test()
    if not success:
        sys.exit(1)
