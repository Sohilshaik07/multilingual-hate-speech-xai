"""
scripts/audit_edge_cases.py

Harness to test all 20 Phase B input edge cases across HECM, Tamil, and Malayalam.
Records:
- expected handling
- actual result
- whether prediction was attempted
- whether UI/pipeline remained stable
- whether clear warning/error was raised
- whether exception escaped
"""

import sys
from pathlib import Path
import os
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ["MODEL_SOURCE"] = "local"  # use local weights for fast audit execution

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from app.utils.model_loader import load_cached_pipeline, predict_sample
from app.utils.labels import get_labels_for_dataset, get_label_name
from app.components.explanations import compute_attention_word_scores, compute_shap_word_scores

EDGE_CASES = [
    ("1. Empty string", ""),
    ("2. Spaces/tabs/newlines", "   \t\n   \r\n  "),
    ("3. One ordinary word", "Namaste"),
    ("4. One punctuation character", "."),
    ("5. Punctuation-only input", "!?...,,:;--!??"),
    ("6. Numbers-only input", "1234567890 987654321"),
    ("7. Emoji-only input", "😡🤬🔥💯👍💔"),
    ("8. URLs and email-like strings", "https://example.com/test?query=hate contact@domain.org"),
    ("9. Repeated tokens", "hate hate hate hate hate hate hate hate hate hate"),
    ("10. Repeated punctuation", "!!!!!!!!!!!!!!!!!!!!????????????????????"),
    ("11. Very short mixed-script text", "hi भाई"),
    ("12. Romanized Hindi-English", "yeh bohot bura aadmi hai stop this"),
    ("13. Devanagari Hindi", "यह समाज के लिए बहुत हानिकारक है"),
    ("14. Tamil script", "இது மிக மோசமான பதிவு உடனடியாக நீக்குங்கள்"),
    ("15. Malayalam script", "ഇത് വളരെ മോശം കാര്യമാണ് എല്ലാവരും ശ്രദ്ധിക്കുക"),
    ("16. English-only text", "This is an entirely English sentence intended to test cross-lingual transfer."),
    ("17. Code-mixed multi-script", "Super video भाई sema mass padam padam"),
    ("18. Unicode combining marks", "Hátè spëëch cömbïnïng \u0300\u0301\u0302\u0303 test"),
    ("19. Zero-width chars / whitespace", "Hate\u200b\u200c\u200dSpeech\u00a0\u2002\u3000Test"),
    ("20. Very long input (>128 tokens)", " ".join(["word" + str(i) for i in range(250)])),
]

def run_audit():
    languages = ["hecm", "tamil", "malayalam"]
    results = {}

    for lang in languages:
        print(f"\n{'='*70}\nAUDITING LANGUAGE: {lang.upper()}\n{'='*70}")
        model, tokenizer, device = load_cached_pipeline(lang, device="cpu")
        num_classes = len(get_labels_for_dataset(lang))
        print(f"Loaded {lang} on {device}. Num classes: {num_classes}")

        lang_results = []
        for case_name, text in EDGE_CASES:
            print(f"\n--- Case: {case_name} ---")
            print(f"Input: {repr(text)[:60]}")
            record = {
                "case": case_name,
                "input": repr(text)[:60],
                "is_empty": not bool(text.strip()),
            }

            # 1. Validation check
            if not text.strip():
                record["validation"] = "REJECTED (Whitespace/Empty)"
                print("  [Validation] Rejected correctly as empty/whitespace.")
                lang_results.append(record)
                continue

            # 2. Prediction
            try:
                pred = predict_sample(model, tokenizer, text, lang, device="cpu")
                record["pred_attempted"] = True
                record["predicted_class"] = pred["predicted_class"]
                record["confidence"] = pred["confidence"]
                record["token_count"] = pred["token_count"]
                record["visible_len"] = len(pred["visible_text"])
                probs = pred["probabilities"]
                
                # Invariants check
                assert len(probs) == num_classes, f"Class count mismatch! {len(probs)} vs {num_classes}"
                assert np.all(np.isfinite(list(probs.values()))), "Non-finite probability!"
                assert abs(sum(probs.values()) - 1.0) < 1e-4, f"Prob sum != 1: {sum(probs.values())}"
                assert pred["predicted_class"] in range(num_classes), "Invalid predicted class index!"
                assert pred["token_count"] <= 128, f"Token count exceeds 128: {pred['token_count']}"
                
                print(f"  [Prediction] Class={pred['predicted_class']} ({get_label_name(lang, pred['predicted_class'])}) "
                      f"Conf={pred['confidence']*100:.2f}% Tokens={pred['token_count']}/128")
                record["pred_status"] = "OK"
            except Exception as e:
                print(f"  [Prediction ERROR] {e}")
                record["pred_status"] = f"ERROR: {e}"
                record["pred_attempted"] = False

            # 3. Attention
            try:
                vis_text = pred.get("visible_text", text)
                attn = compute_attention_word_scores(model, tokenizer, vis_text, device="cpu")
                record["attn_len"] = len(attn)
                assert all(np.isfinite([s for _, s in attn])), "Non-finite attention!"
                print(f"  [Attention] Extracted {len(attn)} word scores. Top: {attn[0] if attn else 'EMPTY'}")
                record["attn_status"] = "OK"
            except Exception as e:
                print(f"  [Attention ERROR] {e}")
                record["attn_status"] = f"ERROR: {e}"

            # 4. SHAP (sample check)
            # Only test SHAP on a representative subset of cases or all cases? Let's test all!
            try:
                vis_text = pred.get("visible_text", text)
                t_cls = pred.get("predicted_class", 0)
                shap_res = compute_shap_word_scores(model, tokenizer, vis_text, t_cls, device="cpu")
                record["shap_len"] = len(shap_res)
                assert all(np.isfinite([s for _, s in shap_res])), "Non-finite SHAP!"
                print(f"  [SHAP] Extracted {len(shap_res)} word scores. Top: {shap_res[0] if shap_res else 'EMPTY'}")
                record["shap_status"] = "OK"
            except Exception as e:
                print(f"  [SHAP ERROR] {e}")
                record["shap_status"] = f"ERROR: {e}"

            lang_results.append(record)

        results[lang] = lang_results

    return results

if __name__ == "__main__":
    run_audit()
