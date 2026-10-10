"""
tests/test_app.py

Unit tests for the Streamlit demonstration interface (app package).
Tests module imports, label configurations, inference shapes, and probability bounds.
"""

import unittest
from pathlib import Path
import numpy as np

# Ensure project root is reachable
import sys
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.utils.labels import (
    LABEL_MAPPINGS,
    get_labels_for_dataset,
    get_label_name,
    get_samples_for_dataset,
)
from app.utils.model_loader import (
    load_cached_pipeline,
    predict_sample,
    get_checkpoint_storage_info,
)
from app.components.explanations import compute_attention_word_scores
from src.models.model_registry import get_checkpoint_dir, CHECKPOINT_PATHS


class TestStreamlitApp(unittest.TestCase):
    """Test suite for Phase 7 Streamlit demonstration application."""

    def test_app_modules_import(self):
        """Verifies that all app components and utilities import cleanly."""
        import app.utils.labels
        import app.utils.model_loader
        import app.components.charts
        import app.components.prediction
        import app.components.explanations
        self.assertTrue(True)

    def test_supported_languages(self):
        """Verifies that all three authoritative language settings are recognized."""
        expected = {"hecm", "tamil", "malayalam"}
        self.assertEqual(set(LABEL_MAPPINGS.keys()), expected)

    def test_label_mappings_counts(self):
        """Verifies correct class counts per task."""
        self.assertEqual(len(LABEL_MAPPINGS["hecm"]), 2)
        self.assertEqual(len(LABEL_MAPPINGS["tamil"]), 6)
        self.assertEqual(len(LABEL_MAPPINGS["malayalam"]), 5)

        # Check specific names
        self.assertEqual(get_label_name("hecm", 0), "Non-Hate")
        self.assertEqual(get_label_name("hecm", 1), "Hate")
        self.assertEqual(get_label_name("malayalam", 0), "Not_offensive")

    def test_checkpoint_paths_and_safetensors_exist(self):
        """Verifies all three model checkpoints exist and contain weights."""
        for key in ["hecm", "tamil", "malayalam"]:
            ckpt_dir = Path(get_checkpoint_dir(key))
            self.assertTrue(ckpt_dir.is_dir(), f"Checkpoint missing: {ckpt_dir}")
            weights_file = ckpt_dir / "model.safetensors"
            self.assertTrue(weights_file.is_file(), f"model.safetensors missing: {weights_file}")

    def test_prediction_output_shape_and_probability_sum(self):
        """Verifies model prediction output structure and probability sum to ~1.0."""
        model, tokenizer, device = load_cached_pipeline("hecm", device="cpu")
        sample_text = "Yeh ek test sentence hai."
        
        result = predict_sample(model, tokenizer, sample_text, "hecm", device="cpu")
        
        self.assertIn("predicted_class", result)
        self.assertIn("confidence", result)
        self.assertIn("probabilities", result)
        self.assertIn("token_count", result)
        
        pred_cls = result["predicted_class"]
        self.assertIn(pred_cls, [0, 1])
        
        probs = result["probabilities"]
        self.assertEqual(len(probs), 2)
        
        prob_sum = sum(probs.values())
        self.assertAlmostEqual(prob_sum, 1.0, places=4)
        self.assertTrue(0.0 <= result["confidence"] <= 1.0)
        self.assertTrue(result["token_count"] <= 128)

    def test_attention_scores_structure(self):
        """Verifies that attention extraction returns valid word-score tuples."""
        model, tokenizer, device = load_cached_pipeline("hecm", device="cpu")
        sample_text = "Modi pagal ho gye hain"
        
        scores = compute_attention_word_scores(model, tokenizer, sample_text, device="cpu")
        self.assertTrue(isinstance(scores, list))
        self.assertTrue(len(scores) > 0)
        for word, score in scores:
            self.assertTrue(isinstance(word, str))
            self.assertTrue(isinstance(score, float))
            self.assertTrue(score >= 0.0)

    def test_sample_presets_presence(self):
        """Verifies that representative samples are configured for each language."""
        for key in ["hecm", "tamil", "malayalam"]:
            samples = get_samples_for_dataset(key)
            self.assertTrue(len(samples) >= 3, f"Expected at least 3 samples for {key}")
            for s in samples:
                self.assertIn("title", s)
                self.assertIn("text", s)
                self.assertTrue(len(s["text"].strip()) > 0)

    def test_checkpoint_storage_info(self):
        """Verifies storage footprint calculation."""
        info = get_checkpoint_storage_info()
        self.assertTrue(info["total_size_gb"] > 1.8)
        self.assertTrue(info["hecm"]["exists"])
        self.assertTrue(info["tamil"]["exists"])
        self.assertTrue(info["malayalam"]["exists"])


if __name__ == "__main__":
    unittest.main()
