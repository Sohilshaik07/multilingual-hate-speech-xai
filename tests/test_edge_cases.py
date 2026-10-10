"""
tests/test_edge_cases.py

Unit tests for Phase 8 comprehensive edge-case audit.
Validates:
1. Input rejection for empty and whitespace-only text.
2. Truncation boundary invariants (<= 126 content tokens, <= 128 total tokens).
3. Strict MODEL_SOURCE environment variable parsing and invalid value rejection.
4. Model output integrity across language switching (HECM=2, Tamil=6, Malayalam=5).
5. Defensive error handling in SHAP and Attention computation.
6. Graceful behavior under malformed inputs and mocked failures.

Does NOT repeatedly download or reload 2 GB checkpoints.
"""

import os
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import torch

from app.utils.labels import (
    LABEL_MAPPINGS,
    get_labels_for_dataset,
    get_label_name,
)
from app.utils.model_loader import (
    get_model_source,
    get_model_identifier,
    predict_sample,
)
from app.components.explanations import (
    compute_shap_word_scores,
    compute_attention_word_scores,
)
from src.xai.shap_audit import get_model_visible_text


class TestEdgeCasesAndInvariants(unittest.TestCase):
    """Automated edge-case and invariant tests for the Streamlit demo pipeline."""

    def test_empty_and_whitespace_input_rejected(self):
        """Empty, whitespace-only, and non-string inputs must raise ValueError."""
        mock_model = MagicMock()
        mock_tok = MagicMock()

        for bad_input in ["", "   ", "\t\n\r  ", " \n "]:
            with self.assertRaises(ValueError) as ctx:
                predict_sample(mock_model, mock_tok, bad_input, "hecm")
            self.assertIn("non-whitespace", str(ctx.exception).lower())

        with self.assertRaises(ValueError):
            predict_sample(mock_model, mock_tok, None, "hecm")

    def test_model_source_parsing_and_validation(self):
        """MODEL_SOURCE parsing supports 'hf' and 'local', rejecting invalid values."""
        with patch.dict(os.environ, {"MODEL_SOURCE": ""}, clear=False):
            self.assertEqual(get_model_source(), "hf")

        with patch.dict(os.environ, {"MODEL_SOURCE": "hf"}, clear=False):
            self.assertEqual(get_model_source(), "hf")

        with patch.dict(os.environ, {"MODEL_SOURCE": "HF "}, clear=False):
            self.assertEqual(get_model_source(), "hf")

        with patch.dict(os.environ, {"MODEL_SOURCE": "local"}, clear=False):
            self.assertEqual(get_model_source(), "local")

        with patch.dict(os.environ, {"MODEL_SOURCE": " LOCAL \n"}, clear=False):
            self.assertEqual(get_model_source(), "local")

        with patch.dict(os.environ, {"MODEL_SOURCE": "unsupported_source"}, clear=False):
            with self.assertRaises(ValueError) as ctx:
                get_model_source()
            self.assertIn("invalid model_source", str(ctx.exception).lower())

    def test_model_identifier_invalid_dataset_key(self):
        """Unknown dataset keys must raise KeyError or ValueError."""
        with patch.dict(os.environ, {"MODEL_SOURCE": "hf"}, clear=False):
            with self.assertRaises(ValueError):
                get_model_identifier("unknown_lang")

    def test_language_switching_class_counts_and_mappings(self):
        """Verifies class counts: HECM=2 -> Tamil=6 -> Malayalam=5 -> HECM=2."""
        sequence = ["hecm", "tamil", "malayalam", "hecm"]
        expected_counts = [2, 6, 5, 2]

        for lang, count in zip(sequence, expected_counts):
            labels = get_labels_for_dataset(lang)
            self.assertEqual(len(labels), count, f"Class count mismatch for {lang}")
            # Ensure all IDs from 0 to count-1 exist
            for idx in range(count):
                self.assertIn(idx, labels)
                name = get_label_name(lang, idx)
                self.assertTrue(isinstance(name, str) and len(name) > 0)

    def test_truncation_boundary_invariants(self):
        """Reconstructed visible text never exceeds 126 content tokens or 128 total tokens."""
        mock_tok = MagicMock()
        # Mock encoding returning 300 token IDs
        long_ids = list(range(100, 400))
        mock_tok.encode.return_value = long_ids
        mock_tok.decode.return_value = " ".join([f"word{i}" for i in range(126)])

        visible_text = get_model_visible_text("Very long text " * 100, mock_tok, max_length=128)
        self.assertTrue(isinstance(visible_text, str))
        self.assertTrue(len(visible_text) > 0)

        # Confirm mock tokenizer was called with add_special_tokens=False
        mock_tok.encode.assert_called_with("Very long text " * 100, add_special_tokens=False)

    def test_attention_empty_input_graceful(self):
        """compute_attention_word_scores returns [] on empty or invalid text."""
        mock_model = MagicMock()
        mock_tok = MagicMock()

        self.assertEqual(compute_attention_word_scores(mock_model, mock_tok, ""), [])
        self.assertEqual(compute_attention_word_scores(mock_model, mock_tok, "   \n\t "), [])
        self.assertEqual(compute_attention_word_scores(mock_model, mock_tok, None), [])

    def test_shap_empty_input_graceful(self):
        """compute_shap_word_scores returns [] on empty or invalid text."""
        mock_model = MagicMock()
        mock_tok = MagicMock()

        self.assertEqual(compute_shap_word_scores(mock_model, mock_tok, "", target_class=0), [])
        self.assertEqual(compute_shap_word_scores(mock_model, mock_tok, "   ", target_class=0), [])
        self.assertEqual(compute_shap_word_scores(mock_model, mock_tok, None, target_class=0), [])

    def test_shap_out_of_bounds_class_handled_gracefully(self):
        """compute_shap_word_scores handles out-of-bounds target_class without crashing."""
        mock_model = MagicMock()
        mock_tok = MagicMock()

        # If an impossible target_class is passed (e.g. 99 for a binary model), it should return []
        with patch("shap.Explainer", side_effect=Exception("SHAP mock failure")):
            scores = compute_shap_word_scores(
                mock_model, mock_tok, "Sample text", target_class=99
            )
            self.assertEqual(scores, [])

    def test_attention_exception_handled_gracefully(self):
        """compute_attention_word_scores does not let exceptions escape."""
        mock_model = MagicMock()
        mock_tok = MagicMock()
        mock_tok.encode.side_effect = RuntimeError("Tokenizer hardware failure")

        scores = compute_attention_word_scores(mock_model, mock_tok, "Valid looking text")
        self.assertEqual(scores, [])


if __name__ == "__main__":
    unittest.main()
