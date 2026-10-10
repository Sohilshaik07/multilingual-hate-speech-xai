# Artifact Manifest — Transferred Experimental Artifacts

> **Authoritative Experimental Record**  
> "The final experimental artifacts were generated and validated in Google Colab. The local repository now contains the transferred authoritative outputs and inference checkpoints. Training-resume state was intentionally excluded."

---

## 1. Overview and Verification Summary

- **Total Transferred Files**: 38 files (excluding pre-existing Phase 1 exploratory data analysis artifacts in `results/eda/`).
- **Total Directory Artifacts in `results/`**: 53 files across 6 major domains (`calibration/`, `eda/`, `final/`, `modeling/`, `robustness/`, `xai/`).
- **Total Final Model Checkpoints**: 3 (`mbert_hecm`, `mbert_tamil`, `mbert_malayalam`).
- **Total Model Checkpoint Size**: 2,143,154,974 bytes (~2,043.87 MB / 2.04 GB).
- **Verification Status**: All files verified for structural integrity, schema adherence, and metric parity. Inference smoke tests confirmed successful classification capability on CPU for all three checkpoints.

---

## 2. Leakage-Safe Splits

| File Path | Type | Records / Split Partition | Authoritative | Locally Reproducible | Purpose & Notes |
|---|---|---|---|---|---|
| `results/modeling/splits/hecm_split_seed42.json` | JSON | `train`: 3,295<br>`val`: 824<br>`test`: 824 | Yes | Yes (Deterministic indices) | Leakage-safe, stratified train/validation/test index partition for Hindi-English code-mixed dataset (HECM). |
| `results/modeling/splits/tamil_split_seed42.json` | JSON | `train`: 15,372<br>`val`: 3,843<br>`test`: 3,844 | Yes | Yes (Deterministic indices) | Stratified train/validation/test index split for Tamil-English code-mixed dataset (DravidianCodeMix). |
| `results/modeling/splits/malayalam_split_seed42.json` | JSON | `train`: 7,022<br>`val`: 1,756<br>`test`: 1,756 | Yes | Yes (Deterministic indices) | Stratified train/validation/test index split for Malayalam-English code-mixed dataset (DravidianCodeMix). |

*Note: These splits enforce strict leakage prevention across all subsequent modeling, calibration, and explainability audits.*

---

## 3. Trained Model Checkpoints (mBERT)

Training-resume state files (`optimizer.pt`, `scheduler.pt`, `scaler.pt`, `rng_state.pth`) were intentionally excluded during transfer to optimize disk storage while preserving full inference capability.

### 3.1. HECM Checkpoint (`results/modeling/mbert_hecm/checkpoint-824/`)
- **Architecture**: `BertForSequenceClassification` (`bert-base-multilingual-cased`)
- **Num Labels**: 2 (Binary: Non-hate vs Hate)
- **Validation Metrics**: Accuracy: 0.814735, Macro-F1: 0.769557, Weighted-F1: 0.810140
- **Total Checkpoint Size**: ~714.37 MB

| File | Size (Bytes) | Type | Purpose | Authoritative | Reproducible |
|---|---|---|---|---|---|
| `config.json` | 883 | JSON | Model architecture and label mapping configuration | Yes | Inference-only |
| `model.safetensors` | 711,443,432 | Binary | Validated mBERT weights in safe tensors format | Yes | Inference-only |
| `tokenizer.json` | 2,919,627 | JSON | Fast tokenizer vocabulary and BPE tokenizer model | Yes | Deterministic |
| `tokenizer_config.json` | 352 | JSON | Tokenizer settings and special token mappings | Yes | Deterministic |
| `trainer_state.json` | 6,290 | JSON | Step-level training logs, loss trajectories, and best epoch | Yes | Historical log |
| `training_args.bin` | 5,265 | Binary | Hugging Face TrainingArguments serialized state | Yes | Historical log |

### 3.2. Tamil Checkpoint (`results/modeling/mbert_tamil/checkpoint-3844/`)
- **Architecture**: `BertForSequenceClassification` (`bert-base-multilingual-cased`)
- **Num Labels**: 6 (Multiclass)
- **Validation Metrics**: Accuracy: 0.710055, Macro-F1: 0.469392, Weighted-F1: 0.731160
- **Total Checkpoint Size**: ~714.39 MB

| File | Size (Bytes) | Type | Purpose | Authoritative | Reproducible |
|---|---|---|---|---|---|
| `config.json` | 1,151 | JSON | Model architecture and 6-class label mapping configuration | Yes | Inference-only |
| `model.safetensors` | 711,455,736 | Binary | Validated mBERT weights in safe tensors format | Yes | Inference-only |
| `tokenizer.json` | 2,919,627 | JSON | Fast tokenizer vocabulary and BPE tokenizer model | Yes | Deterministic |
| `tokenizer_config.json` | 352 | JSON | Tokenizer settings and special token mappings | Yes | Deterministic |
| `trainer_state.json` | 9,175 | JSON | Step-level training logs, loss trajectories, and best epoch | Yes | Historical log |
| `training_args.bin` | 5,265 | Binary | Hugging Face TrainingArguments serialized state | Yes | Historical log |

### 3.3. Malayalam Checkpoint (`results/modeling/mbert_malayalam/checkpoint-1756/`)
- **Architecture**: `BertForSequenceClassification` (`bert-base-multilingual-cased`)
- **Num Labels**: 5 (Multiclass)
- **Validation Metrics**: Accuracy: 0.918919, Macro-F1: 0.450386, Weighted-F1: 0.908449
- **Total Checkpoint Size**: ~714.38 MB

| File | Size (Bytes) | Type | Purpose | Authoritative | Reproducible |
|---|---|---|---|---|---|
| `config.json` | 1,271 | JSON | Model architecture and 5-class label mapping configuration | Yes | Inference-only |
| `model.safetensors` | 711,452,660 | Binary | Validated mBERT weights in safe tensors format | Yes | Inference-only |
| `tokenizer.json` | 2,919,627 | JSON | Fast tokenizer vocabulary and BPE tokenizer model | Yes | Deterministic |
| `tokenizer_config.json` | 352 | JSON | Tokenizer settings and special token mappings | Yes | Deterministic |
| `trainer_state.json` | 8,644 | JSON | Step-level training logs, loss trajectories, and best epoch | Yes | Historical log |
| `training_args.bin` | 5,265 | Binary | Hugging Face TrainingArguments serialized state | Yes | Historical log |

---

## 4. Explainable AI (XAI) Audit Artifacts

| File Path | Type | Purpose | Authoritative | Locally Reproducible | Known Limitations & Notes |
|---|---|---|---|---|---|
| `results/xai/hecm_attention_shap_agreement.csv` | CSV | Top-k word agreement between self-attention and KernelSHAP on HECM test set | Yes | No (requires expensive Colab GPU compute) | Aggregated agreement metrics; Top-word agreement is 0.23. |
| `results/xai/hecm_attention_shap_rank_correlation.csv` | CSV | Sample-level Spearman rank correlation between attention weights and SHAP attribution for HECM | Yes | No | Mean Spearman: 0.3128, Median Spearman: 0.3316. |
| `results/xai/malayalam_shap_100_audit.csv` | CSV | Token-level attributions and perturbation confidence drops across 100 audited Malayalam test samples | Yes | No | Authoritative sample-level audit table. (Note: Corresponding raw 100-sample CSV audits for HECM and Tamil were not generated/transferred; their downstream summaries are authoritative). |
| `results/xai/malayalam_xai_summary.json` | JSON | Summary faithfulness metrics (faithfulness drops, p-values, rank correlation) for Malayalam | Yes | No | Contains exact test statistics (Attention Top vs Random p: 0.0797, SHAP drop: 0.0658). |
| `results/xai/malayalam_attention_shap_agreement.csv` | CSV | Top-k word agreement between self-attention and KernelSHAP for Malayalam | Yes | No | Top-word agreement is 0.26. |
| `results/xai/malayalam_attention_shap_rank_correlation.csv` | CSV | Sample-level Spearman rank correlation between attention and SHAP for Malayalam | Yes | No | Mean Spearman: 0.1677, Median Spearman: 0.2443. |
| `results/xai/tamil_xai_summary_final.json` | JSON | Summary faithfulness metrics (faithfulness drops, p-values, rank correlation) for Tamil | Yes | No | Top-word agreement: 0.18; Mean Spearman: 0.1858; Attention Top vs Random p: 0.6209. |

---

## 5. Model Calibration Artifacts

| File Path | Type | Purpose | Authoritative | Locally Reproducible | Known Limitations & Notes |
|---|---|---|---|---|---|
| `results/calibration/calibration_results.csv` | CSV | Cross-language Expected Calibration Error (ECE), Brier Score, and Mean Confidence | Yes | Yes (from checkpoint outputs) | Authoritative calibration table matching project spec: HECM ECE=0.071284, Tamil ECE=0.066790, Malayalam ECE=0.065863. |
| `results/calibration/hecm_reliability_bins.csv` | CSV | 10-bin confidence vs empirical accuracy statistics for HECM test split | Yes | Yes | Used to generate reliability diagram. |
| `results/calibration/tamil_reliability_bins.csv` | CSV | 10-bin confidence vs empirical accuracy statistics for Tamil test split | Yes | Yes | Used to generate reliability diagram. |
| `results/calibration/malayalam_reliability_bins.csv` | CSV | 10-bin confidence vs empirical accuracy statistics for Malayalam test split | Yes | Yes | Used to generate reliability diagram. |
| `results/calibration/hecm_reliability_diagram.png` | PNG | Visual reliability diagram and calibration curve for HECM | Yes | Yes | Transferred reference diagram. |
| `results/calibration/tamil_reliability_diagram.png` | PNG | Visual reliability diagram and calibration curve for Tamil | Yes | Yes | Transferred reference diagram. |
| `results/calibration/malayalam_reliability_diagram.png` | PNG | Visual reliability diagram and calibration curve for Malayalam | Yes | Yes | Transferred reference diagram. |

---

## 6. Robustness and Perturbation Artifacts

| File Path | Type | Purpose | Authoritative | Locally Reproducible | Known Limitations & Notes |
|---|---|---|---|---|---|
| `results/robustness/formatting_robustness_results.csv` | CSV | Predictive performance across punctuation removal, casing alteration, and whitespace noise | Yes | No (original transformation pipeline executed in Colab) | Authoritative record of formatting resilience across all 3 languages. |
| `results/robustness/script_composition_per_class_metrics.csv` | CSV | Fine-grained performance broken down by script composition (Native vs Romanized vs Mixed-Script) | Yes | No | Documents severe performance drops on mixed-script tokens (e.g. Malayalam mixed-script Macro-F1: 0.335658). |

---

## 7. Master Experimental Results Table

| File Path | Type | Purpose | Authoritative | Locally Reproducible | Known Limitations & Notes |
|---|---|---|---|---|---|
| `results/final/master_experimental_results.csv` | CSV | Definitive summary consolidating predictive, calibration, faithfulness, correlation, and mixed-script metrics across HECM, Tamil, and Malayalam | Yes | Yes (Consolidation of authoritative component metrics) | Authoritative single-source-of-truth table for Phase 3 reporting and publication figures. |
