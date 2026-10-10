"""
scripts/verify_artifacts.py

Verification script auditing all experimental artifacts, directories,
checkpoints, splits, and XAI outputs transferred from Google Colab.

Strict Principles:
- Read-only inspection; NEVER modifies or overwrites any artifact.
- Validates presence, non-zero file sizes, and expected internal keys.
"""

import os
import sys
import json
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def verify_artifacts() -> bool:
    print("=" * 65)
    print("TEAM 6 XAI — COMPREHENSIVE ARTIFACT AUDIT")
    print("=" * 65)

    all_passed = True

    # 1. Directory Tree Verification
    expected_dirs = [
        "results/modeling/splits",
        "results/modeling/mbert_hecm/checkpoint-824",
        "results/modeling/mbert_tamil/checkpoint-3844",
        "results/modeling/mbert_malayalam/checkpoint-1756",
        "results/xai",
        "results/calibration",
        "results/robustness",
        "results/final",
        "results/eda"
    ]
    print("\n[1] Checking Directory Structure:")
    for d in expected_dirs:
        p = os.path.join(PROJECT_ROOT, d)
        exists = os.path.isdir(p)
        status = "PASS" if exists else "FAIL"
        print(f"  [{status}] {d}")
        if not exists:
            all_passed = False

    # 2. Checkpoint Files Verification
    ckpt_files = [
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "trainer_state.json",
        "training_args.bin"
    ]
    ckpts = {
        "HECM": "results/modeling/mbert_hecm/checkpoint-824",
        "Tamil": "results/modeling/mbert_tamil/checkpoint-3844",
        "Malayalam": "results/modeling/mbert_malayalam/checkpoint-1756"
    }
    print("\n[2] Checking Final Model Checkpoints:")
    for name, rel in ckpts.items():
        print(f"  --- {name} ({rel}) ---")
        for f in ckpt_files:
            fp = os.path.join(PROJECT_ROOT, rel, f)
            exists = os.path.isfile(fp) and os.path.getsize(fp) > 0
            status = "PASS" if exists else "FAIL"
            sz = os.path.getsize(fp) if exists else 0
            print(f"    [{status}] {f:<22} ({sz:,} bytes)")
            if not exists:
                all_passed = False

    # 3. Splits Verification
    splits = {
        "HECM": ("results/modeling/splits/hecm_split_seed42.json", 1398),
        "Tamil": ("results/modeling/splits/tamil_split_seed42.json", 6584),
        "Malayalam": ("results/modeling/splits/malayalam_split_seed42.json", 2997)
    }
    print("\n[3] Checking Leakage-Safe Splits:")
    for name, (rel, exp_test) in splits.items():
        fp = os.path.join(PROJECT_ROOT, rel)
        exists = os.path.isfile(fp) and os.path.getsize(fp) > 0
        if exists:
            with open(fp, "r", encoding="utf-8") as f:
                d = json.load(f)
            test_cnt = len(d["indices"]["test"]) if "indices" in d else len(d["test_indices"])
            matches = (test_cnt == exp_test)
            status = "PASS" if matches else "FAIL"
            print(f"  [{status}] {name:<10} Test Count: {test_cnt} (Expected: {exp_test})")
            if not matches:
                all_passed = False
        else:
            print(f"  [FAIL] {name:<10} File Missing: {rel}")
            all_passed = False

    # 4. Calibration Artifacts
    calib_files = [
        "results/calibration/calibration_results.csv",
        "results/calibration/hecm_reliability_bins.csv",
        "results/calibration/tamil_reliability_bins.csv",
        "results/calibration/malayalam_reliability_bins.csv",
        "results/calibration/hecm_reliability_diagram.png",
        "results/calibration/tamil_reliability_diagram.png",
        "results/calibration/malayalam_reliability_diagram.png"
    ]
    print("\n[4] Checking Calibration Artifacts:")
    for rel in calib_files:
        fp = os.path.join(PROJECT_ROOT, rel)
        exists = os.path.isfile(fp) and os.path.getsize(fp) > 0
        status = "PASS" if exists else "FAIL"
        print(f"  [{status}] {rel}")
        if not exists:
            all_passed = False

    # 5. Robustness Artifacts
    rob_files = [
        "results/robustness/formatting_robustness_results.csv",
        "results/robustness/script_composition_per_class_metrics.csv"
    ]
    print("\n[5] Checking Robustness Artifacts:")
    for rel in rob_files:
        fp = os.path.join(PROJECT_ROOT, rel)
        exists = os.path.isfile(fp) and os.path.getsize(fp) > 0
        status = "PASS" if exists else "FAIL"
        print(f"  [{status}] {rel}")
        if not exists:
            all_passed = False

    # 6. XAI Artifacts
    xai_files = [
        "results/xai/hecm_attention_shap_agreement.csv",
        "results/xai/hecm_attention_shap_rank_correlation.csv",
        "results/xai/malayalam_shap_100_audit.csv",
        "results/xai/malayalam_xai_summary.json",
        "results/xai/malayalam_attention_shap_agreement.csv",
        "results/xai/malayalam_attention_shap_rank_correlation.csv",
        "results/xai/tamil_xai_summary_final.json"
    ]
    print("\n[6] Checking XAI Artifacts:")
    for rel in xai_files:
        fp = os.path.join(PROJECT_ROOT, rel)
        exists = os.path.isfile(fp) and os.path.getsize(fp) > 0
        status = "PASS" if exists else "FAIL"
        print(f"  [{status}] {rel}")
        if not exists:
            all_passed = False

    # 7. Final Master Table
    master_path = os.path.join(PROJECT_ROOT, "results/final/master_experimental_results.csv")
    print("\n[7] Checking Master Results Table:")
    exists = os.path.isfile(master_path) and os.path.getsize(master_path) > 0
    status = "PASS" if exists else "FAIL"
    print(f"  [{status}] results/final/master_experimental_results.csv")
    if not exists:
        all_passed = False

    print("\n" + "=" * 65)
    print(f"ARTIFACT AUDIT OVERALL RESULT: {'ALL PASS' if all_passed else 'FAILURES DETECTED'}")
    print("=" * 65)
    return all_passed


if __name__ == "__main__":
    success = verify_artifacts()
    sys.exit(0 if success else 1)
