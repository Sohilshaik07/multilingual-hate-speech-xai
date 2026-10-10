"""
tests/test_loaders.py

Tests for dataset loading, split integrity, and leakage prevention.
"""

import unittest
from src.data.loaders import (
    load_raw_dataset,
    load_split_indices,
    validate_split_integrity,
    load_split_data
)


class TestDataLoaders(unittest.TestCase):

    def test_raw_dataset_shapes(self):
        """Verifies total sample counts of raw datasets."""
        df_hecm = load_raw_dataset("hecm")
        self.assertEqual(len(df_hecm), 9357)

        df_tamil = load_raw_dataset("tamil")
        self.assertEqual(len(df_tamil), 43919)

        df_malayalam = load_raw_dataset("malayalam")
        self.assertEqual(len(df_malayalam), 20010)

    def test_split_indices_structure(self):
        """Verifies that split JSONs return 'train', 'validation', and 'test' keys."""
        for key in ["hecm", "tamil", "malayalam"]:
            indices = load_split_indices(key)
            self.assertIn("train", indices)
            self.assertIn("validation", indices)
            self.assertIn("test", indices)
            self.assertTrue(len(indices["train"]) > 0)
            self.assertTrue(len(indices["validation"]) > 0)
            self.assertTrue(len(indices["test"]) > 0)

    def test_split_integrity_zero_leakage(self):
        """Verifies zero cross-partition leakage across all three datasets."""
        for key in ["hecm", "tamil", "malayalam"]:
            report = validate_split_integrity(key)
            self.assertTrue(report["zero_index_leakage"], f"Index leakage detected in {key}")
            self.assertTrue(report["counts_match"], f"Sample count mismatch in {key}")
            self.assertEqual(report["text_train_test_overlap"], 0, f"Text leakage detected in {key}")
            self.assertTrue(report["integrity_passed"])

    def test_load_split_data_slice(self):
        """Verifies that split data loading returns correct subset length."""
        test_df = load_split_data("hecm", "test")
        self.assertEqual(len(test_df), 1398)
        self.assertIn("text", test_df.columns)
        self.assertIn("label", test_df.columns)


if __name__ == "__main__":
    unittest.main()
