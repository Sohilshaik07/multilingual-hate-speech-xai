"""
tests/test_paths.py

Tests for canonical project paths and directory resolution.
"""

import os
import unittest
from configs.paths import (
    PROJECT_ROOT,
    DATASETS_DIR,
    RESULTS_DIR,
    SPLITS_DIR,
    CALIBRATION_DIR,
    ROBUSTNESS_DIR,
    XAI_DIR,
    FINAL_DIR
)


class TestPaths(unittest.TestCase):

    def test_core_directories_exist(self):
        """Verifies that all primary project directories exist."""
        for d in [PROJECT_ROOT, DATASETS_DIR, RESULTS_DIR, SPLITS_DIR, CALIBRATION_DIR, ROBUSTNESS_DIR, XAI_DIR, FINAL_DIR]:
            self.assertTrue(os.path.isdir(d), f"Directory does not exist: {d}")


if __name__ == "__main__":
    unittest.main()
