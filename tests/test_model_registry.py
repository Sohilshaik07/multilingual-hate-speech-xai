"""
tests/test_model_registry.py

Tests for model registry, label mappings, and checkpoint path resolution.
"""

import os
import unittest
from src.models.model_registry import (
    get_model_spec,
    get_checkpoint_dir,
    load_tokenizer,
    NUM_LABELS,
    HECM_LABEL2ID,
    TAMIL_LABEL2ID,
    MALAYALAM_LABEL2ID
)


class TestModelRegistry(unittest.TestCase):

    def test_model_specs(self):
        """Verifies model specs for all three datasets."""
        for key in ["hecm", "tamil", "malayalam"]:
            spec = get_model_spec(key)
            self.assertEqual(spec["dataset_key"], key)
            self.assertEqual(spec["num_labels"], NUM_LABELS[key])
            self.assertEqual(spec["max_seq_length"], 128)
            self.assertTrue(os.path.isdir(spec["checkpoint_dir"]))

    def test_checkpoint_paths_exist(self):
        """Verifies that all three checkpoint directories exist locally."""
        for key in ["hecm", "tamil", "malayalam"]:
            ckpt_dir = get_checkpoint_dir(key)
            self.assertTrue(os.path.isdir(ckpt_dir), f"Missing checkpoint dir: {ckpt_dir}")
            self.assertTrue(os.path.isfile(os.path.join(ckpt_dir, "model.safetensors")))
            self.assertTrue(os.path.isfile(os.path.join(ckpt_dir, "config.json")))

    def test_label_mappings(self):
        """Verifies label mappings match project definitions."""
        self.assertEqual(HECM_LABEL2ID, {"0": 0, "1": 1})
        self.assertEqual(len(TAMIL_LABEL2ID), 6)
        self.assertEqual(len(MALAYALAM_LABEL2ID), 5)
        self.assertEqual(MALAYALAM_LABEL2ID["Not_offensive"], 0)

    def test_tokenizer_loading(self):
        """Lightweight test to ensure tokenizers load successfully."""
        tok = load_tokenizer("hecm")
        encoded = tok("Test sentence", return_tensors="pt")
        self.assertIn("input_ids", encoded)


if __name__ == "__main__":
    unittest.main()
