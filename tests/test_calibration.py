"""
tests/test_calibration.py

Tests for calibration module (ECE, Brier score, reliability bins).
"""

import unittest
import numpy as np
from src.calibration.calibration import (
    extract_confidences,
    compute_brier_score,
    compute_reliability_bins,
    compute_ece_from_bins,
    compute_calibration_summary
)


class TestCalibration(unittest.TestCase):

    def test_extract_confidences(self):
        probs = np.array([
            [0.8, 0.2],
            [0.3, 0.7],
            [0.6, 0.4]
        ])
        confs, preds = extract_confidences(probs)
        np.testing.assert_allclose(confs, [0.8, 0.7, 0.6])
        np.testing.assert_array_equal(preds, [0, 1, 0])

    def test_brier_score_perfect(self):
        # Perfect predictions have Brier score 0
        probs = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ])
        y_true = [0, 1]
        brier = compute_brier_score(probs, y_true)
        self.assertAlmostEqual(brier, 0.0)

    def test_brier_score_worst(self):
        # Completely wrong predictions have Brier score 2.0 (for 2 classes)
        probs = np.array([
            [0.0, 1.0],
            [1.0, 0.0]
        ])
        y_true = [0, 1]
        brier = compute_brier_score(probs, y_true)
        self.assertAlmostEqual(brier, 2.0)

    def test_reliability_bins_structure(self):
        probs = np.array([
            [0.9, 0.1],
            [0.8, 0.2],
            [0.2, 0.8]
        ])
        y_true = [0, 0, 1]
        bins_df = compute_reliability_bins(y_true, probs, num_bins=15)
        self.assertEqual(len(bins_df), 15)
        self.assertListEqual(list(bins_df.columns), ["bin", "lower", "upper", "count", "mean_confidence", "accuracy", "gap"])
        self.assertEqual(bins_df["count"].sum(), 3)

    def test_ece_from_bins(self):
        probs = np.array([
            [0.9, 0.1],
            [0.8, 0.2],
            [0.2, 0.8]
        ])
        y_true = [0, 0, 1]
        summary = compute_calibration_summary(y_true, probs, num_bins=15)
        self.assertIn("ece", summary)
        self.assertIn("brier_score", summary)
        self.assertTrue(0.0 <= summary["ece"] <= 1.0)


if __name__ == "__main__":
    unittest.main()
