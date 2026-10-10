# Streamlit Deployment Guide

**Project:** Faithfulness-Aware Explainability Audit of Multilingual Transformer Models for Code-Mixed Hate Speech Detection  
**Team:** Team 6  
**App entry point:** pp/app.py

---

## 1. Architecture Overview

`
Team6_XAI/
├── app/
│   ├── app.py                  # Streamlit UI entry point
│   ├── utils/
│   │   ├── model_loader.py     # Model/tokenizer loader — HF Hub or local
│   │   └── labels.py           # Label mappings and sample texts
│   └── components/
│       ├── explanations.py     # SHAP and Attention computation + rendering
│       ├── prediction.py       # Prediction result panel
│       └── charts.py           # Plotly/matplotlib chart helpers
├── src/
│   ├── models/model_registry.py   # Authoritative checkpoint paths
│   ├── xai/shap_audit.py          # Truncation-safe text masking
│   └── xai/attention_audit.py     # Attention score extraction
├── .streamlit/config.toml         # Streamlit server + theme config
├── requirements_app.txt           # Minimal cloud deployment requirements
└── requirements.txt               # Full research environment requirements
`

### Dual Model Source Architecture

The app selects its model source via the MODEL_SOURCE environment variable:

| MODEL_SOURCE | Source | Use case |
|---|---|---|
| hf (default) | HuggingFace Hub | Streamlit Community Cloud, any cloud host |
| local | Local checkpoint directories | Development, local research execution |

---

## 2. HuggingFace Model Repositories

All three validated mBERT inference checkpoints are hosted publicly on HuggingFace Hub:

| Language Pair | HF Repository | Classes |
|---|---|---|
| Hindi-English (HECM) | sohilshaik07/team6-mbert-hecm | Binary (2) |
| Tamil-English | sohilshaik07/team6-mbert-tamil | Multiclass (6) |
| Malayalam-English | sohilshaik07/team6-mbert-malayalam | Multiclass (5) |

Each repository contains the complete 7-file model package including model.safetensors (~694 MB).

---

## 3. Running Locally

### Option A — Local Checkpoints
`powershell
 = "local"
streamlit run app/app.py
`

### Option B — HuggingFace Hub (test cloud path locally)
`powershell
 = "hf"
streamlit run app/app.py
`

---

## 4. Streamlit Community Cloud Deployment

1. Push repository to GitHub (model weights excluded by .gitignore)
2. Sign in to share.streamlit.io
3. Create New App:
   - Main file: app/app.py
   - Requirements: requirements_app.txt
4. Add Secret: MODEL_SOURCE = "hf"
5. Deploy. First cold start downloads weights from HF Hub (~8-15 min per language selected).

---

## 5. Docker Deployment

`dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements_app.txt
EXPOSE 8501
ENV MODEL_SOURCE=hf
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.headless=true"]
`

---

## 6. Technical Notes

- **Attention:** Load with attn_implementation='eager' for output_attentions=True support
- **Sequence length:** 128 max tokens (126 content + [CLS] + [SEP])
- **SHAP version:** shap==0.52.0 pinned to match Colab research environment
- **Device:** CPU by default; GPU used automatically if torch.cuda.is_available()

---

## 7. Research Integrity Constraints

- No model retraining (inference-only)
- No modification of experimental CSV/JSON artifacts
- No causal faithfulness claims (behavioral relevance framing only)
- Aggregate audit results displayed from fixed validated baseline
- SHAP not claimed as ground-truth explanation
- Attention failure qualified per-language, not universal claim

---

*Phase 7 — Streamlit Research Demonstration Interface. Research results are frozen and authoritative.*
