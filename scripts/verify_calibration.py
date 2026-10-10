"""
scripts/verify_calibration.py

Verifies calibration evaluation metrics, Expected Calibration Error (ECE),
Brier scores, and 15-bin reliability calculations against authoritative CSV artifacts.

Strict Principles:
- Read-only inspection; NEVER overwrites authoritative artifacts.
- Validates ECE calculation from raw bin tables.
- Verification outputs are saved separately under results/verification/.
"""

import os
import sys
import json
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.calibration.calibration import compute_ece_from_bins

CALIBRATION_CSV = os.path.join(PROJECT_ROOT, "results", "calibration", "calibration_results.csv")
VERIFICATION_DIR = os.path.join(PROJECT_ROOT, "results", "verification")

BIN_FILES = {
    "HECM": os.path.join(PROJECT_ROOT, "results", "calibration", "hecm_reliability_bins.csv"),
    "Tamil": os.path.join(PROJECT_ROOT, "results", "calibration", "tamil_reliability_bins.csv"),
    "Malayalam": os.path.join(PROJECT_ROOT, "results", "calibration", "malayalam_reliability_bins.csv")
}

EXPECTED_ECE = {
    "HECM": 0.071284,
    "Tamil": 0.066790,
    "Malayalam": 0.065863
}

EXPECTED_BRIER = {
    "HECM": 0.282710,
    "Tamil": 0.393545,
    "Malayalam": 0.146696
}

TOLERANCE = 1e-4


def verify_calibration() -> bool:
    print("=" * 70)
    print("TEAM 6 XAI — CALIBRATION METRICS PARITY AUDIT")
    print("=" * 70)

    if not os.path.exists(CALIBRATION_CSV):
        print(f"[FAIL] Calibration summary CSV not found at: {CALIBRATION_CSV}")
        return False

    df_calib = pd.read_csv(CALIBRATION_CSV)
    print(f"Loaded calibration summary table:\n{df_calib.to_string()}\n")

    all_passed = True
    audit_results = {}

    for name, bin_path in BIN_FILES.items():
        print(f"--- Checking {name} Reliability Bins ---")
        if not os.path.exists(bin_path):
            print(f"  [FAIL] Missing bin file: {bin_path}")
            all_passed = False
            continue

        bins_df = pd.read_csv(bin_path)
        recomputed_ece = compute_ece_from_bins(bins_df)
        expected_ece = EXPECTED_ECE[name]
        diff_ece = abs(recomputed_ece - expected_ece)
        ece_passed = diff_ece <= TOLERANCE

        # Check against calibration_results.csv row
        row = df_calib[df_calib["Language"] == name]
        if len(row) > 0:
            csv_ece = float(row["ECE"].values[0])
            csv_brier = float(row["Brier_Score"].values[0])
            csv_conf = float(row["Mean_Confidence"].values[0])
            expected_brier = EXPECTED_BRIER[name]
            diff_brier = abs(csv_brier - expected_brier)
            brier_passed = diff_brier <= TOLERANCE
        else:
            csv_ece, csv_brier, csv_conf = np.nan, np.nan, np.nan
            brier_passed = False

        status = "PASS" if (ece_passed and brier_passed) else "FAIL"
        print(f"  [{status}] Recomputed ECE: {recomputed_ece:.6f} | Target: {expected_ece:.6f} | Diff: {diff_ece:.2e}")
        print(f"  [{status}] Recorded Brier:   {csv_brier:.6f} | Target: {expected_brier:.6f} | Diff: {diff_brier:.2e}")
        print(f"  [INFO] Total Bin Sample Count: {bins_df['count'].sum()}")

        if not (ece_passed and brier_passed):
            all_passed = False

        audit_results[name] = {
            "recomputed_ece": float(recomputed_ece),
            "expected_ece": float(expected_ece),
            "diff_ece": float(diff_ece),
            "recorded_brier": float(csv_brier),
            "expected_brier": float(expected_brier),
            "diff_brier": float(diff_brier),
            "status": status
        }

    # Save non-destructive verification log
    os.makedirs(VERIFICATION_DIR, exist_ok=True)
    report_path = os.path.join(VERIFICATION_DIR, "calibration_verification.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "overall_passed": all_passed,
            "tolerance": TOLERANCE,
            "results": audit_results
        }, f, indent=2)
    print(f"\nSaved calibration verification log to: {report_path}")

    print("=" * 70)
    print(f"CALIBRATION AUDIT OVERALL RESULT: {'ALL PASS' if all_passed else 'FAILURES DETECTED'}")
    print("=" * 70)
    return all_passed


if __name__ == "__main__":
    success = verify_calibration()
    sys.exit(0 if success else 1)
