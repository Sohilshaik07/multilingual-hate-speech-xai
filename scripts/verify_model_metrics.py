"""
scripts/verify_model_metrics.py

Verifies model performance metrics against authoritative experimental records
and validated project specifications.

Strict Principles:
- Read-only inspection; NEVER overwrites authoritative artifacts.
- Verifies accuracy, macro-F1, weighted-F1, calibration, and agreement.
- Verification outputs are saved separately under results/verification/.
"""

import os
import sys
import json
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MASTER_CSV = os.path.join(PROJECT_ROOT, "results", "final", "master_experimental_results.csv")
VERIFICATION_DIR = os.path.join(PROJECT_ROOT, "results", "verification")

# Authoritative specification targets
TARGET_METRICS = {
    "HECM": {
        "Accuracy": 0.814735,
        "Macro-F1": 0.769557,
        "Weighted-F1": 0.810140,
        "ECE": 0.071284,
        "Brier": 0.282710,
        "Top_Word_Agreement": 0.23,
        "Mixed_Script_N": 76,
        "Mixed_Script_Accuracy": 0.815789,
        "Mixed_Script_Macro_F1": 0.675214
    },
    "Tamil": {
        "Accuracy": 0.710055,
        "Macro-F1": 0.469392,
        "Weighted-F1": 0.731160,
        "ECE": 0.066790,
        "Brier": 0.393545,
        "Top_Word_Agreement": 0.18,
        "Mixed_Script_N": 273,
        "Mixed_Script_Accuracy": 0.739927,
        "Mixed_Script_Macro_F1": 0.452456
    },
    "Malayalam": {
        "Accuracy": 0.918919,
        "Macro-F1": 0.450386,
        "Weighted-F1": 0.908449,
        "ECE": 0.065863,
        "Brier": 0.146696,
        "Top_Word_Agreement": 0.26,
        "Mixed_Script_N": 413,
        "Mixed_Script_Accuracy": 0.961259,
        "Mixed_Script_Macro_F1": 0.335658
    }
}

TOLERANCE = 1e-4


def verify_metrics() -> bool:
    print("=" * 70)
    print("TEAM 6 XAI — MODEL METRICS PARITY AUDIT")
    print("=" * 70)

    if not os.path.exists(MASTER_CSV):
        print(f"[FAIL] Master CSV not found at: {MASTER_CSV}")
        return False

    df_master = pd.read_csv(MASTER_CSV)
    print(f"Loaded master results table: {len(df_master)} rows.")

    all_passed = True
    audit_results = {}

    for _, row in df_master.iterrows():
        lang = str(row["Language"])
        if lang not in TARGET_METRICS:
            print(f"[WARN] Unknown language in master CSV: {lang}")
            continue

        print(f"\n--- Checking Language: {lang} ---")
        targets = TARGET_METRICS[lang]
        audit_results[lang] = {}

        for metric, target_val in targets.items():
            if metric not in row:
                print(f"  [FAIL] Metric '{metric}' missing from master table!")
                all_passed = False
                audit_results[lang][metric] = {"status": "MISSING"}
                continue

            actual_val = row[metric]
            diff = abs(actual_val - target_val)
            passed = diff <= TOLERANCE
            status = "PASS" if passed else "FAIL"

            print(f"  [{status}] {metric:<25} Actual: {actual_val:<10.6f} Target: {target_val:<10.6f} Diff: {diff:.2e}")
            if not passed:
                all_passed = False

            audit_results[lang][metric] = {
                "actual": float(actual_val),
                "target": float(target_val),
                "diff": float(diff),
                "status": status
            }

    # Save verification audit output (non-destructive)
    os.makedirs(VERIFICATION_DIR, exist_ok=True)
    report_path = os.path.join(VERIFICATION_DIR, "model_metrics_verification.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "overall_passed": all_passed,
            "tolerance": TOLERANCE,
            "results": audit_results
        }, f, indent=2)
    print(f"\nSaved verification log to: {report_path}")

    print("=" * 70)
    print(f"MODEL METRICS AUDIT OVERALL RESULT: {'ALL PASS' if all_passed else 'FAILURES DETECTED'}")
    print("=" * 70)
    return all_passed


if __name__ == "__main__":
    success = verify_metrics()
    sys.exit(0 if success else 1)
