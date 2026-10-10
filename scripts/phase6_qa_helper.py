"""Phase 6 QA helper: checkpoint audit + security scan."""
import re, json
from pathlib import Path

ROOT = Path(r"C:\Users\SKSOH\OneDrive\Desktop\Team6_XAI")

# === CHECKPOINT AUDIT ===
print("=== CHECKPOINT AUDIT ===")
chk_map = {
    "HECM":      ROOT / "results/modeling/mbert_hecm/checkpoint-824",
    "Tamil":     ROOT / "results/modeling/mbert_tamil/checkpoint-3844",
    "Malayalam": ROOT / "results/modeling/mbert_malayalam/checkpoint-1756",
}
training_state_names = {"optimizer.pt","scheduler.pt","rng_state.pth",
                        "scaler.pt","random_states_0.pkl","amp_optimizer_state.pt"}
for lang, chk in chk_map.items():
    if not chk.exists():
        print(f"  MISSING: {lang} checkpoint {chk}")
        continue
    files = sorted([f.name for f in chk.iterdir()])
    print(f"\n{lang} ({chk.name}):")
    for f in files:
        size_kb = (chk/f).stat().st_size / 1024
        flag = "  ⚠ TRAINING-STATE" if f in training_state_names else ""
        print(f"    {f:45s} {size_kb:>10.1f} KB{flag}")

# trainer_state.json is metadata only (not optimizer weights) — note it
print("\nNOTE: trainer_state.json = training history metadata only (no optimizer weights). Safe to keep.")
print("NOTE: training_args.bin = serialized TrainingArguments. Not optimizer state. Safe to keep.")

# === SECURITY AUDIT ===
print("\n=== SECURITY AUDIT ===")
secret_patterns = [
    (r"hf_[A-Za-z0-9]{20,}", "HuggingFace token"),
    (r"AKIA[0-9A-Z]{16}", "AWS access key"),
    (r"sk-[A-Za-z0-9]{32,}", "OpenAI key"),
    (r"(?i)api[_\s]?key\s*[=:]\s*[\"']?[A-Za-z0-9\-_]{20,}", "Generic API key"),
    (r"(?i)password\s*=\s*[\"'][^\"']{6,}", "Hardcoded password"),
    (r"ghp_[A-Za-z0-9]{36}", "GitHub PAT"),
]
scan_exts = {".py", ".yaml", ".yml", ".json", ".env", ".txt", ".md", ".toml", ".cfg", ".ini"}
skip_substrings = ["__pycache__", ".git", "model.safetensors",
                   "tokenizer.json", "tokenizer_config.json",
                   "trainer_state.json", "training_args.bin", "config.json",
                   "Team6_XAI_validated_artifacts.zip"]
findings = []
for f in ROOT.rglob("*"):
    if not f.is_file():
        continue
    if f.suffix not in scan_exts:
        continue
    if any(s in str(f) for s in skip_substrings):
        continue
    try:
        content = f.read_text(encoding="utf-8", errors="ignore")
        for pat, label in secret_patterns:
            m = re.search(pat, content)
            if m:
                findings.append((str(f.relative_to(ROOT)), label))
    except Exception:
        pass

# Also check for .env files
for f in ROOT.rglob(".env"):
    findings.append((str(f.relative_to(ROOT)), ".env file"))
for f in ROOT.rglob("secrets.toml"):
    findings.append((str(f.relative_to(ROOT)), "secrets.toml"))

if findings:
    for path, label in findings:
        print(f"  POSSIBLE SECRET [{label}]: {path}")
else:
    print("  No secrets or credential files detected.")

# === PDF QA ===
print("\n=== PDF QA ===")
try:
    import fitz
    pdf_path = ROOT / "reports" / "final_report.pdf"
    doc = fitz.open(str(pdf_path))
    print(f"  Pages: {len(doc)}")
    # Check first page for title
    p0_text = doc[0].get_text()
    has_title = "Faithfulness" in p0_text and "Explainability" in p0_text
    print(f"  Title on cover page: {'YES' if has_title else 'NO'}")
    # Check for required section headings across all pages
    all_text = "\n".join(doc[i].get_text() for i in range(len(doc)))
    required = ["Abstract", "Introduction", "Related Work", "Datasets",
                "Methodology", "Experimental Setup", "Predictive Performance",
                "Calibration", "Explainability Audit", "Robustness",
                "Discussion", "Limitations", "Conclusion", "References"]
    for sec in required:
        found = sec in all_text
        print(f"  Section '{sec}': {'FOUND' if found else 'MISSING'}")
    # Check for broken characters / encoding artefacts
    garbled = len(re.findall(r"[\ufffd\x00-\x08\x0e-\x1f]", all_text))
    print(f"  Garbled/null chars: {garbled}")
    # Check page sizes
    sizes = set()
    for i in range(len(doc)):
        r = doc[i].rect
        sizes.add((round(r.width), round(r.height)))
    print(f"  Page sizes (pts): {sizes}")
    doc.close()
except Exception as e:
    print(f"  PDF QA error: {e}")

# === FIGURE QA ===
print("\n=== FIGURE QA ===")
fig_dir = ROOT / "figures"
figs = [f"fig{i:02d}" for i in range(1, 11)]
all_ok = True
for stem in figs:
    # Find matching file
    matches = list(fig_dir.glob(f"{stem}_*.png"))
    png_ok = len(matches) == 1
    pdf_matches = list(fig_dir.glob(f"{stem}_*.pdf"))
    pdf_ok = len(pdf_matches) == 1
    if png_ok:
        size_kb = matches[0].stat().st_size / 1024
        status = "OK" if size_kb > 50 else "SUSPICIOUSLY SMALL"
    else:
        size_kb = 0
        status = "MISSING"
    print(f"  {stem}: PNG={'OK' if png_ok else 'MISSING'} ({size_kb:.0f}KB)  PDF={'OK' if pdf_ok else 'MISSING'}  [{status}]")
    if not (png_ok and pdf_ok):
        all_ok = False
print(f"  All 10 figures present: {'YES' if all_ok else 'NO'}")
