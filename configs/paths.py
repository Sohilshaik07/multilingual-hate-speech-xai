"""
configs/paths.py

Canonical path constants for the Team6_XAI project.

All path references in scripts and modules should import from here
to avoid hardcoded strings scattered across the codebase.

Usage:
    from configs.paths import RESULTS_DIR, CALIBRATION_RESULTS_CSV
"""

import os

# ============================================================
# PROJECT ROOT
# Resolve relative to this file's location (configs/ is one level
# below the project root).
# ============================================================
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_THIS_DIR)


# ============================================================
# DATASETS
# ============================================================
DATASETS_DIR        = os.path.join(PROJECT_ROOT, "datasets")
HINDI_DIR           = os.path.join(DATASETS_DIR, "Hindi")
TAMIL_DIR           = os.path.join(DATASETS_DIR, "TAMIL")
MALAYALAM_DIR       = os.path.join(DATASETS_DIR, "MALAYALAM")
HATECHECKIN_DIR     = os.path.join(DATASETS_DIR, "HateCheckHIn")

HECM_TEXT_FILE      = os.path.join(HINDI_DIR, "HECM.txt")
HECM_CLASS_FILE     = os.path.join(HINDI_DIR, "HECM_Class.txt")


# ============================================================
# RESULTS — top-level
# ============================================================
RESULTS_DIR         = os.path.join(PROJECT_ROOT, "results")

# EDA outputs (Review-1 artifacts — DO NOT overwrite)
EDA_DIR             = os.path.join(RESULTS_DIR, "eda")
EDA_FIGURES_DIR     = os.path.join(EDA_DIR, "figures")
EDA_TABLES_DIR      = os.path.join(EDA_DIR, "tables")
EDA_SUMMARY_MD      = os.path.join(EDA_DIR, "REVIEW1_EDA_SUMMARY.md")

# Modeling splits and checkpoints
MODELING_DIR        = os.path.join(RESULTS_DIR, "modeling")
SPLITS_DIR          = os.path.join(MODELING_DIR, "splits")

HECM_SPLIT_JSON     = os.path.join(SPLITS_DIR, "hecm_split_seed42.json")
TAMIL_SPLIT_JSON    = os.path.join(SPLITS_DIR, "tamil_split_seed42.json")
MALAYALAM_SPLIT_JSON= os.path.join(SPLITS_DIR, "malayalam_split_seed42.json")

MBERT_HECM_DIR      = os.path.join(MODELING_DIR, "mbert_hecm")
MBERT_TAMIL_DIR     = os.path.join(MODELING_DIR, "mbert_tamil")
MBERT_MALAYALAM_DIR = os.path.join(MODELING_DIR, "mbert_malayalam")

# Calibration outputs
CALIBRATION_DIR     = os.path.join(RESULTS_DIR, "calibration")
CALIBRATION_CSV     = os.path.join(CALIBRATION_DIR, "calibration_results.csv")

# XAI outputs
XAI_DIR             = os.path.join(RESULTS_DIR, "xai")

# Robustness outputs
ROBUSTNESS_DIR      = os.path.join(RESULTS_DIR, "robustness")
FORMATTING_ROBUSTNESS_CSV   = os.path.join(ROBUSTNESS_DIR, "formatting_robustness_results.csv")
SCRIPT_COMPOSITION_CSV      = os.path.join(ROBUSTNESS_DIR, "script_composition_per_class_metrics.csv")

# Final / master results
FINAL_DIR           = os.path.join(RESULTS_DIR, "final")
MASTER_RESULTS_CSV  = os.path.join(FINAL_DIR, "master_experimental_results.csv")


# ============================================================
# FIGURES (publication-ready, post-modeling)
# ============================================================
FIGURES_DIR         = os.path.join(PROJECT_ROOT, "figures")


# ============================================================
# REPORTS
# ============================================================
REPORTS_DIR         = os.path.join(PROJECT_ROOT, "reports")


# ============================================================
# NOTEBOOKS
# ============================================================
NOTEBOOKS_DIR       = os.path.join(PROJECT_ROOT, "notebooks")


# ============================================================
# SCRIPTS
# ============================================================
SCRIPTS_DIR         = os.path.join(PROJECT_ROOT, "scripts")
