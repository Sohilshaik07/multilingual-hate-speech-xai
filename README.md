# Faithfulness-Aware Explainability Audit of Multilingual Transformer Models for Code-Mixed Hate Speech Detection

**Team 6 — Final-Year Research Project**

---

## 1. Research Objective

This project evaluates the faithfulness, trustworthiness, calibration, and robustness of explainability methods (SHAP and Self-Attention) applied to multilingual transformer models fine-tuned for hate speech detection in code-mixed social media text across three Indian language pairs:
1. **Hindi-English (HECM)** — paired text/label files
2. **Tamil-English (DravidianCodeMix)** — multiclass classification
3. **Malayalam-English (DravidianCodeMix)** — multiclass classification

### Core Research Questions
- **Deletion-based Faithfulness**: Do SHAP token attributions exhibit statistically significant behavioral relevance when the most important tokens are removed?
- **Self-Attention Faithfulness**: Do transformer self-attention weights from the final layer exhibit comparable behavioral relevance, or do they fail faithfulness criteria?
- **Explainability Agreement**: What is the degree of agreement (top-token overlap and Spearman rank correlation) between feature attribution via SHAP and Self-Attention?
- **Model Calibration**: How well-calibrated are the predictive posterior probabilities across languages (Expected Calibration Error [ECE], Brier Score, and reliability diagrams)?
- **Robustness & Script Composition**: How resilient are model predictions to surface formatting variations and Unicode script-composition differences (Native Indic vs. Latin vs. Mixed-Script)?

---

## 2. Languages & Datasets

| Language Pair | Dataset | Total Samples | Leakage-Safe Test Size | Primary Script |
|---|---|---|---|---|
| **Hindi-English** | HECM | 9,357 | 1,398 | Devanagari & Latin |
| **Tamil-English** | DravidianCodeMix | 43,919 | 6,584 | Tamil & Latin |
| **Malayalam-English** | DravidianCodeMix | 20,010 | 2,997 | Malayalam & Latin |

Raw datasets are stored in `datasets/` and are strictly **immutable**. Preprocessing applies Unicode NFC normalization, zero-width space removal, and standardized mention/URL tokens without destructive filtering or row deletion.

---

## 3. Models & Checkpoints

- **Selected Architecture**: Multilingual BERT (`bert-base-multilingual-cased` / `mBERT`), fine-tuned with `max_seq_length=128`.
- **Authoritative Checkpoints**:
  - HECM: `results/modeling/mbert_hecm/checkpoint-824/` (Binary classification, 2 labels)
  - Tamil: `results/modeling/mbert_tamil/checkpoint-3844/` (Multiclass, 6 labels)
  - Malayalam: `results/modeling/mbert_malayalam/checkpoint-1756/` (Multiclass, 5 labels)

> **Important**: Model checkpoints transferred to this repository are **inference-only checkpoints** (containing `model.safetensors`, `config.json`, and tokenizer configurations). Training-resume state (`optimizer.pt`, `scheduler.pt`) was intentionally omitted to optimize repository size. **No model training is performed locally.**

---

## 4. Leakage-Safe Evaluation Protocol

All splits were created using a **text-group-aware stratified strategy** (`seed=42`, target ratio 70/15/15) to guarantee:
- Zero cross-partition index overlap (`train_test = 0`, `val_test = 0`)
- Zero text duplicate leakage across partitions
- Assignment of all conflicting duplicate groups strictly to training

Split indices are stored deterministically in:
- `results/modeling/splits/hecm_split_seed42.json`
- `results/modeling/splits/tamil_split_seed42.json`
- `results/modeling/splits/malayalam_split_seed42.json`

---

## 5. Authoritative Target Metrics

The validated experimental findings generated and validated in Google Colab are:

| Metric | HECM (Hindi-English) | Tamil-English | Malayalam-English |
|---|---|---|---|
| **Accuracy** | 0.814735 | 0.710055 | 0.918919 |
| **Macro-F1** | 0.769557 | 0.469392 | 0.450386 |
| **Weighted-F1** | 0.810140 | 0.731160 | 0.908449 |
| **ECE (15 bins)** | 0.071284 | 0.066790 | 0.065863 |
| **Brier Score** | 0.282710 | 0.393545 | 0.146696 |
| **Mean Confidence** | 0.882042 | 0.776845 | 0.984075 |
| **Top-Word Agreement** | 0.23 (23%) | 0.18 (18%) | 0.26 (26%) |
| **Mean Spearman Rho** | 0.312823 | 0.185783 | 0.167700 |
| **SHAP Drop Top vs Random ($p$)** | 0.001134 (Sig) | 0.000220 (Sig) | 0.000245 (Sig) |
| **Attention Drop Top vs Random ($p$)** | 0.156692 (Non-sig) | 0.620854 (Non-sig) | 0.079719 (Marginal) |
| **Mixed-Script Support ($N$)** | 76 | 273 | 413 |
| **Mixed-Script Macro-F1** | 0.675214 | 0.452456 | 0.335658 |

---

## 6. Distinguishing Historical Outputs vs. Local Reproducibility

### A. Validated Historical Experimental Outputs (Authoritative)
The experimental CSVs, JSONs, and checkpoint weights in `results/` represent the authoritative record generated in Google Colab under GPU acceleration. These files must **never be overwritten, regenerated, or fabricated locally**.

### B. Local Reproducibility Code (`src/` & `scripts/`)
The local codebase implements modular, testable software engineering that encapsulates the complete research methodology:
- Model registry and inference loading (`src/models/model_registry.py`)
- Split and dataset loaders with leakage checks (`src/data/loaders.py`)
- Standardized classification evaluation (`src/evaluation/metrics.py`)
- Calibration metrics, ECE binning, and Brier calculations (`src/calibration/calibration.py`)
- SHAP and attention deletion auditing algorithms (`src/xai/`)
- Cross-method agreement and Spearman correlation (`src/xai/agreement.py`)
- Robustness and Unicode script composition analyzers (`src/robustness/`)

---

## 7. Known Methodological Limitations

1. **Script Composition vs. Linguistic Code-Mixing**:
   Script composition is evaluated via Unicode blocks (Latin vs Native Indic). Transliterated text (e.g., Romanized Hindi/Tamil/Malayalam) is represented entirely in the Latin alphabet and is categorized as single-script, underestimating true linguistic code-mixing.
2. **Formatting Transformations**:
   The original synthetic perturbation generators were executed in Colab and are not locally recoverable; `formatting_robustness_results.csv` serves as the authoritative historical baseline.
3. **Hardware & Compute Scope**:
   Full 100-sample KernelSHAP audits require substantial GPU compute and are intended for batch execution, not local lightweight testing.

---

---

## 8. Repository Layout & Deliverables

```
Team6_XAI/
├── reports/
│   ├── final_report.md           # Full academic research report (Markdown)
│   ├── final_report.pdf          # 16-page publication-quality PDF report
│   ├── REPORT_VALIDATION.md      # Numerical and citation verification certificate
│   └── PHASE6_FINAL_QA.md        # Comprehensive submission QA audit
├── figures/                      # 10 publication figures (300 DPI PNG & PDF)
│   ├── fig01_predictive_performance.{png,pdf}
│   ├── fig02_calibration_ece.{png,pdf}
│   ├── fig03_reliability_hecm.{png,pdf}
│   ├── fig04_reliability_tamil.{png,pdf}
│   ├── fig05_reliability_malayalam.{png,pdf}
│   ├── fig06_shap_faithfulness.{png,pdf}
│   ├── fig07_attention_faithfulness.{png,pdf}
│   ├── fig08_attention_shap_agreement.{png,pdf}
│   ├── fig09_formatting_robustness.{png,pdf}
│   └── fig10_script_composition.{png,pdf}
├── datasets/                     # Immutable raw benchmarks (HECM, Tamil, Malayalam)
├── results/                      # Authoritative historical experimental artifacts
│   ├── final/                    # master_experimental_results.csv
│   ├── calibration/              # ECE, Brier, reliability diagram bins
│   ├── xai/                      # SHAP, Attention audits & rank correlations
│   ├── robustness/               # Formatting robustness & script composition
│   ├── modeling/                 # Checkpoints & leakage-safe splits
│   └── eda/                      # Exploratory data analysis summaries & figures
├── src/                          # Modular reproducible research library
├── scripts/                      # Verification, parity audit, and figure generators
├── tests/                        # 23 automated unit and regression tests
├── configs/                      # Experiment parameters and path registry
└── requirements.txt              # Complete dependency specifications
```

---

## 9. Verification & Test Execution

Run the suite of automated integrity tests and verification scripts:

```bash
# 1. Run all unit tests (23/23 tests pass)
python -m pytest tests/ -v

# 2. Verify all transferred artifact paths, checkpoints, and files
python scripts/verify_artifacts.py

# 3. Verify model predictive and faithfulness metrics against master records
python scripts/verify_model_metrics.py

# 4. Verify calibration metrics (ECE, Brier, 15-bin distributions)
python scripts/verify_calibration.py

# 5. Verify exact report-to-artifact numerical parity
python scripts/audit_parity.py

# 6. Verify scientific claim boundaries and absence of overclaims
python scripts/audit_claims.py

# 7. Run full end-to-end smoke test for Streamlit demo
python scripts/smoke_test_app.py
```

---

## 10. Interactive Demonstration Application

### Running the Demo Locally

Launch the research demonstration interface using Streamlit:

```bash
streamlit run app/app.py
```

### Core Interface Features
- **Language & Model Switcher:** Switch between fine-tuned mBERT models for HECM (Hindi-English), Tamil-English, and Malayalam-English.
- **Model Inference & Posterior Probabilities:** Real-time softmax probability distributions and confidence scores under truncation-safe boundaries (126 content tokens).
- **Word-Level KernelSHAP Attributions:** Inline highlighted text and horizontal contribution charts showing tokens supporting or opposing the predicted category.
- **Self-Attention Importance:** Final-layer (Layer 12) multi-head attention concentration weights toward the classification token.
- **Side-by-Side Comparison:** Direct ranking comparison between SHAP and Attention highlights method divergence.
- **Authoritative Benchmark Registry:** Interactive browser for all 10 publication figures and validated research findings.

### Crucial Methodological Distinction
- **Authoritative Research Findings:** The quantitative statistics reported in `reports/final_report.md` (e.g., ECE, Macro-F1, Wilcoxon $p$-values, Spearman $\rho$) are fixed historical outputs computed over the entire 73,286-instance benchmark in Google Colab.
- **Interactive Demonstration:** The demo UI provides illustrative, single-sample qualitative analysis. Predictions and explanations generated in the UI do not alter or manufacture new research findings.

### Deployment Scope & Limitations
- **Local Execution:** Supported out of the box using local PyTorch CPU inference without requiring CUDA.
- **Checkpoint Footprint:** The three mBERT checkpoints require **2.00 GB** of local storage (`model.safetensors` files are ~694 MB each).
- **Streamlit Community Cloud Limitation:** Because GitHub enforces a strict 100 MB per-file upload limit, the 694 MB weights cannot be stored in a vanilla GitHub repository. Deployment to Streamlit Cloud requires hosting model checkpoints on an external artifact store (such as Hugging Face Model Hub or AWS S3).


