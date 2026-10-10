"""
tests/test_metrics.py

Tests for evaluation metrics functions.
"""

import unittest
import numpy as np
from src.evaluation.metrics import (
    compute_accuracy,
    compute_macro_f1,
    compute_weighted_f1,
    compute_per_class_metrics,
    compute_confusion_matrix,
    compute_all_metrics
)


class TestMetrics(unittest.TestCase):

    def setUp(self):
        self.y_true = [0, 1, 0, 1, 1, 0, 1, 0]
        self.y_pred = [0, 1, 0, 0, 1, 0, 1, 1]

    def test_accuracy(self):
        acc = compute_accuracy(self.y_true, self.y_pred)
        # 6 correct out of 8 = 0.75
        self.assertAlmostEqual(acc, 0.75)

    def test_macro_and_weighted_f1(self):
        macro_f1 = compute_macro_f1(self.y_true, self.y_pred)
        weighted_f1 = compute_weighted_f1(self.y_true, self.y_pred)
        self.assertTrue(0.0 <= macro_f1 <= 1.0)
        self.assertTrue(0.0 <= weighted_f1 <= 1.0)

    def test_per_class_metrics(self):
        df_per_class = compute_per_class_metrics(self.y_true, self.y_pred, labels=[0, 1])
        self.assertEqual(len(df_per_class), 2)
        self.assertListEqual(list(df_per_class.columns), ["class", "precision", "recall", "f1", "support"])

    def test_confusion_matrix(self):
        cm = compute_confusion_matrix(self.y_true, self.y_pred, labels=[0, 1])
        self.assertEqual(cm.shape, (2, 2))
        self.assertEqual(cm.sum(), 8)

    def test_compute_all_metrics(self):
        res = compute_all_metrics(self.y_true, self.y_pred, labels=[0, 1])
        self.assertIn("accuracy", res)
        self.assertIn("macro_f1", res)
        self.assertIn("weighted_f1", res)
        self.assertIn("per_class", res)
        self.assertIn("confusion_matrix", res)


if __name__ == "__main__":
    unittest.main()
