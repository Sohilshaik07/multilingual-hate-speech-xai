"""
audit_claims.py
===============
Audits reports and documentation for prohibited overclaims.
Verifies careful scientific framing.
"""
import sys, re
from pathlib import Path

# Configure utf-8 stdout
sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"C:\Users\SKSOH\OneDrive\Desktop\Team6_XAI")

def run_claim_audit():
    files_to_check = [
        ROOT / "reports/final_report.md",
        ROOT / "README.md",
        ROOT / "reports/REPORT_VALIDATION.md",
    ]

    forbidden_regexes = [
        (r"(?i)causal(ly)?\s+faithful", "Claims causal faithfulness"),
        (r"(?i)ground[\s-]truth\s+explanation", "Claims ground-truth explanation"),
        (r"(?i)universally\s+superior", "Claims universally superior"),
        (r"(?i)attention\s+is\s+universally\s+unfaithful", "Claims attention is universally unfaithful"),
        (r"(?i)attention\s+can\s+never\s+be", "Claims attention can never be an explanation"),
        (r"(?i)understands\s+code[\s-]mixing", "Claims model understands code-mixing"),
        (r"(?i)robust\s+to\s+code[\s-]mixing", "Claims model is robust to code-mixing"),
        (r"(?i)proves\s+(linguistic\s+)?code[\s-]mixing\s+robustness", "Claims proves code-mixing robustness"),
        (r"(?i)proves\s+general\s+robustness", "Claims proves general robustness"),
        (r"(?i)our\s+new\s+model", "Claims new model"),
        (r"(?i)our\s+novel\s+model", "Claims novel model"),
        (r"(?i)novel\s+dataset", "Claims novel dataset"),
        (r"(?i)we\s+created\s+the\s+dataset", "Claims created dataset"),
    ]

    disclaimer_words = [
        "not", "never", "neither", "distinction", "distinguish", "cannot",
        "caution", "unsupported", "disclaimer", "prohibited",
        "rather than", "instead of", "does not", "without claiming",
        "absent", "should not", "do not", "fails to establish", "no causal"
    ]

    all_passed = True
    for fp in files_to_check:
        txt = fp.read_text(encoding="utf-8")
        print(f"\nChecking {fp.name}:")
        file_clean = True
        for pat, label in forbidden_regexes:
            matches = list(re.finditer(pat, txt))
            for m in matches:
                start = max(0, m.start() - 100)
                end = min(len(txt), m.end() + 100)
                context = txt[start:end].replace("\n", " ").strip()
                is_negation = any(neg in context.lower() for neg in disclaimer_words)
                print(f"  Match [{label}]: \"{context}\" -> Properly Qualified: {is_negation}")
                if not is_negation:
                    file_clean = False
                    all_passed = False
        if file_clean:
            print("  -> PASSED: All occurrences are properly qualified/hedged.")
        else:
            print("  -> WARNING: Found unhedged claim!")

    # Check for presence of required careful framing
    report_text = (ROOT / "reports/final_report.md").read_text(encoding="utf-8")
    framing_found = "SHAP demonstrated statistically significant deletion-based behavioral relevance" in report_text or \
                    "behavioural relevance" in report_text.lower()
    print(f"\nCareful framing 'behavioural relevance' present: {framing_found}")

    diff_distinction = ("behavioural relevance" in report_text.lower()) and \
                       ("causal faithfulness" in report_text.lower()) and \
                       ("ground-truth" in report_text.lower() or "ground truth" in report_text.lower())
    print(f"Distinction between behavioural relevance, causal faithfulness, ground-truth present: {diff_distinction}")

    return all_passed and framing_found and diff_distinction

if __name__ == "__main__":
    success = run_claim_audit()
    print(f"\nOverall Claim Audit: {'PASS' if success else 'FAIL'}")
