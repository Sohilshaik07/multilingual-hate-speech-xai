"""
scripts/run_eda.py

Canonical runner for Exploratory Data Analysis (EDA) and Review-1 pipeline.
Executes the validated preprocessing, feature engineering, and EDA reporting.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from run_review1 import run_pipeline

if __name__ == "__main__":
    run_pipeline()
