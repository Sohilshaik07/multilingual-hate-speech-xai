"""
tests/test_robustness.py

Tests for formatting robustness and script composition analysis modules.
"""

import unittest
from src.robustness.script_composition import (
    classify_character_script,
    classify_text_script_composition,
    load_authoritative_per_class_metrics,
    compute_composition_macro_f1_gap
)
from src.robustness.formatting_robustness import (
    load_formatting_robustness_results,
    summarize_formatting_robustness
)


class TestRobustness(unittest.TestCase):

    def test_character_script_classification(self):
        # Latin
        self.assertEqual(classify_character_script("A", "Hindi-English"), "Latin")
        self.assertEqual(classify_character_script("z", "Tamil-English"), "Latin")

        # Devanagari (Hindi)
        self.assertEqual(classify_character_script("क", "Hindi-English"), "Native")

        # Tamil
        self.assertEqual(classify_character_script("த", "Tamil-English"), "Native")

        # Malayalam
        self.assertEqual(classify_character_script("മ", "Malayalam-English"), "Native")

        # Other
        self.assertEqual(classify_character_script("1", "Hindi-English"), "Other")
        self.assertEqual(classify_character_script(" ", "Tamil-English"), "Other")

    def test_text_script_composition(self):
        # Pure Latin
        self.assertEqual(classify_text_script_composition("Hello world", "Hindi-English"), "Single-script")

        # Pure Native
        self.assertEqual(classify_text_script_composition("नमस्ते भारत", "Hindi-English"), "Single-script")

        # Mixed Script
        self.assertEqual(classify_text_script_composition("Hello नमस्ते", "Hindi-English"), "Mixed-script")

    def test_load_authoritative_formatting_robustness(self):
        df = load_formatting_robustness_results()
        self.assertEqual(len(df), 8)
        self.assertIn("Prediction_Consistency", df.columns)
        self.assertTrue((df["Prediction_Consistency"] == 1.0).all())

        summary = summarize_formatting_robustness()
        self.assertEqual(summary["total_experiments"], 8)
        self.assertEqual(summary["mean_prediction_consistency"], 1.0)

    def test_load_authoritative_script_composition(self):
        df = load_authoritative_per_class_metrics()
        self.assertEqual(len(df), 26)
        self.assertIn("composition", df.columns)

        gaps = compute_composition_macro_f1_gap()
        self.assertIn("HECM", gaps)
        self.assertIn("Tamil", gaps)
        self.assertIn("Malayalam", gaps)


if __name__ == "__main__":
    unittest.main()
