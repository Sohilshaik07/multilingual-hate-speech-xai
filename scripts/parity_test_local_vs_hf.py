"""
scripts/parity_test_local_vs_hf.py

Deterministic Local-vs-HuggingFace Logit/Probability Parity Test.

For each of 3 language checkpoints (HECM, Tamil, Malayalam), runs 3 fixed
inputs through BOTH the local checkpoint and the corresponding HF Hub
repository and reports the maximum absolute difference in softmax probabilities.

Usage:
    python scripts/parity_test_local_vs_hf.py

Constraints:
- Does not modify any model or research artifact.
- Does not retrain anything.
- Reads models in inference-only mode.
"""

import sys
import os
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
import json
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.models.model_registry import (
    get_checkpoint_dir,
    MAX_SEQ_LENGTH,
)

# ---------------------------------------------------------------------------
# HF Hub repository map (verified public repos)
# ---------------------------------------------------------------------------
HF_REPOS = {
    "hecm":      "sohilshaik07/team6-mbert-hecm",
    "tamil":     "sohilshaik07/team6-mbert-tamil",
    "malayalam": "sohilshaik07/team6-mbert-malayalam",
}

# ---------------------------------------------------------------------------
# Fixed test inputs — 3 per language (deterministic, never modified)
# ---------------------------------------------------------------------------
FIXED_INPUTS = {
    "hecm": [
        "Yeh log bahut bure hain aur inhe desh se bahar karna chahiye",
        "I love the diversity in our country it makes us stronger",
        "Ye sab bakwaas hai, koi bhi inhe seriously nahi leta",
    ],
    "tamil": [
        "Avan romba nallavana irukaan",
        "இவர்கள் நம் நாட்டிற்கு எதிரானவர்கள்",
        "This movie was really amazing and entertaining to watch",
    ],
    "malayalam": [
        "ഇത് വളരെ നല്ലതാണ്",
        "These people are a disgrace to society",
        "Malayalam cinema has produced many great films this year",
    ],
}


def load_model_and_tokenizer(model_id: str, device: str = "cpu"):
    """Load model and tokenizer once for inference."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id, attn_implementation="eager"
        )
    except (TypeError, ValueError):
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.eval()
    model.to(torch.device(device))
    return model, tokenizer


def get_probs_from_loaded(model, tokenizer, text: str, device: str = "cpu") -> np.ndarray:
    """Return softmax probabilities using pre-loaded model and tokenizer."""
    inputs = tokenizer(
        text,
        max_length=MAX_SEQ_LENGTH,
        truncation=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(torch.device(device)) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
    return probs


def run_parity_test():
    print("=" * 70)
    print("LOCAL vs HUGGINGFACE HUB DETERMINISTIC PARITY TEST")
    print("=" * 70)
    print()

    results = {}
    all_pass = True
    TOLERANCE = 1e-4  # Max allowable absolute difference (float32 precision)

    for lang_key in ["hecm", "tamil", "malayalam"]:
        local_id = get_checkpoint_dir(lang_key)
        hf_id = HF_REPOS[lang_key]
        inputs = FIXED_INPUTS[lang_key]

        print(f"[{lang_key.upper()}]")
        print(f"  Local path : {local_id}")
        print(f"  HF repo    : {hf_id}")

        print("  Loading local model...")
        local_model, local_tok = load_model_and_tokenizer(local_id)
        print("  Loading HF model...")
        hf_model, hf_tok = load_model_and_tokenizer(hf_id)

        max_abs_diff = 0.0
        per_input_diffs = []

        for i, text in enumerate(inputs):
            print(f"  Input {i+1}: \"{text[:55]}{'...' if len(text) > 55 else ''}\"")

            local_probs = get_probs_from_loaded(local_model, local_tok, text)
            hf_probs = get_probs_from_loaded(hf_model, hf_tok, text)

            abs_diff = np.abs(local_probs - hf_probs)
            max_diff = float(abs_diff.max())
            max_abs_diff = max(max_abs_diff, max_diff)

            per_input_diffs.append(max_diff)

            local_pred = int(np.argmax(local_probs))
            hf_pred = int(np.argmax(hf_probs))
            pred_match = "✓" if local_pred == hf_pred else "✗ MISMATCH"

            print(f"    Local probs : {np.round(local_probs, 6).tolist()}")
            print(f"    HF    probs : {np.round(hf_probs, 6).tolist()}")
            print(f"    Max |diff|  : {max_diff:.2e}  |  Pred: Local={local_pred}, HF={hf_pred}  [{pred_match}]")

        lang_pass = max_abs_diff < TOLERANCE
        all_pass = all_pass and lang_pass
        status = "PASS" if lang_pass else "FAIL"

        print(f"  --> Max absolute probability difference (across 3 inputs): {max_abs_diff:.2e}")
        print(f"  --> Tolerance: {TOLERANCE:.0e}  |  Result: [{status}]")
        print()

        results[lang_key] = {
            "local_id": str(local_id),
            "hf_repo": hf_id,
            "max_abs_diff": max_abs_diff,
            "per_input_max_diffs": per_input_diffs,
            "pass": lang_pass,
        }

    print("=" * 70)
    print("PARITY TEST SUMMARY")
    print("=" * 70)
    for lang_key, r in results.items():
        status = "PASS" if r["pass"] else "FAIL"
        print(f"  {lang_key.upper():<12}  max|diff| = {r['max_abs_diff']:.2e}  [{status}]")
    print()

    overall = "ALL PASS" if all_pass else "FAILURES DETECTED"
    print(f"Overall result: {overall}")
    print("=" * 70)

    # Save results to JSON for record
    out_path = ROOT / "reports" / "parity_test_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: {out_path}")

    return all_pass


if __name__ == "__main__":
    success = run_parity_test()
    sys.exit(0 if success else 1)
