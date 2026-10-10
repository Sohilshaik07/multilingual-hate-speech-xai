# Phase 8 — Comprehensive Streamlit Edge-Case Audit Report

**Project:** Faithfulness-Aware Explainability Audit of Multilingual Transformer Models for Code-Mixed Hate Speech Detection  
**Team:** Team 6  
**Audit Date:** 2026-10-10  
**Audit Status:** **PASS**

---

## 1. Executive Summary

This report documents the comprehensive edge-case audit conducted on the Streamlit demonstration interface (`app/app.py`), its underlying utility layers (`app/utils/`), explainability components (`app/components/`), and model pipelines across all three validated language settings (**HECM Hindi-English**, **Tamil-English DravidianCodeMix**, and **Malayalam-English DravidianCodeMix**).

### Key Audit Metrics
| Metric | Baseline (Phase 7) | Post-Audit (Phase 8) | Status |
| :--- | :---: | :---: | :---: |
| **Unit Test Suite** | 31 / 31 PASSED | **40 / 40 PASSED** | **PASS** |
| **Local Model Smoke Test** | PASSED | **PASSED** | **PASS** |
| **Hugging Face Model Smoke Test** | PASSED | **PASSED** | **PASS** |
| **Local vs HF Probability Parity** | Max diff: 0.00e+00 | **Max diff: 0.00e+00** | **PASS** |
| **Input Edge Cases Audited** | Baseline samples | **20 cases × 3 languages (60 runs)** | **PASS** |
| **Authoritative Research Files** | Verified | **100% Unchanged (Verified SHA256)** | **PASS** |
| **Checkpoint Weights** | 3 safetensors | **100% Unchanged (678.5 MB each)** | **PASS** |

---

## 2. Phase B — Input Edge Cases Audited (20 Cases × 3 Languages)

Each edge case was systematically evaluated across all three language settings.

| # | Edge Case Category | Test Input Representation | Expected Handling | Actual Result | Prediction Attempted | UI & Pipeline State |
| :---: | :--- | :--- | :--- | :--- | :---: | :--- |
| **1** | Empty string | `""` | Reject with clear validation warning; do not classify | Rejected (`ValueError` / Warning) | No | Responsive; cleared stale state |
| **2** | Whitespace-only | `"   \t\n\r  "` | Reject with clear validation warning; do not classify | Rejected (`ValueError` / Warning) | No | Responsive; cleared stale state |
| **3** | One ordinary word | `"Namaste"` | Ingest, pad/encode, produce valid distribution | Classified (`Non-Hate` 95.0%, Tamil C1, Mal C0) | Yes | Normal, finite attributions |
| **4** | One punctuation char | `"."` | Ingest single token, produce valid distribution | Classified (`Hate` 92.5%, Tamil C0, Mal C0) | Yes | Normal, finite attributions |
| **5** | Punctuation-only | `"!?...,,:;--!??"` | Ingest punctuation subwords, valid probabilities | Classified (Finite, sum = 1.0) | Yes | Normal, finite attributions |
| **6** | Numbers-only | `"1234567890 987654321"` | Tokenize numeric subwords, bounded probabilities | Classified (Finite, sum = 1.0) | Yes | Normal, finite attributions |
| **7** | Emoji-only | `"😡🤬🔥💯👍💔"` | Decodes to `[UNK]`, handles without crash | Classified via `[UNK]` token | Yes | Normal, finite attributions |
| **8** | URLs and emails | `"https://example.com/test contact@domain.org"` | Tokenize domain/protocol tokens safely | Classified (Finite, sum = 1.0) | Yes | Normal, finite attributions |
| **9** | Repeated tokens | `"hate hate hate ..."` (10 repetitions) | Handle multi-token sequence and subword grouping | Classified (`Hate` 77.1%, Tamil C4, Mal C0) | Yes | Normal, finite attributions |
| **10** | Repeated punctuation | `"!!!!...????"` (40 chars) | Handle subword expansion without blowing up | Classified (Finite, sum = 1.0) | Yes | Normal, finite attributions |
| **11** | Short mixed-script | `"hi भाई"` | Tokenize Latin + Devanagari subwords | Classified (Finite, sum = 1.0) | Yes | Normal, finite attributions |
| **12** | Romanized Hindi-English| `"yeh bohot bura aadmi hai stop this"` | Valid transliterated classification | Classified (`Hate` 88.6%, Tamil C1, Mal C1) | Yes | Normal, finite attributions |
| **13** | Devanagari Hindi | `"यह समाज के लिए बहुत हानिकारक है"` | Native Devanagari script evaluation | Classified (`Hate` 93.6%, Tamil C0, Mal C1) | Yes | Normal, finite attributions |
| **14** | Tamil script | `"இது மிக மோசமான பதிவு உடனடியாக நீக்குங்கள்"`| Native Tamil script evaluation | Classified (Tamil C0 72.1%, Mal C0) | Yes | Normal, finite attributions |
| **15** | Malayalam script | `"ഇത് വളരെ മോശം കാര്യമാണ്..."` | Native Malayalam script evaluation | Classified (Mal C0 99.5%, Tamil C0) | Yes | Normal, finite attributions |
| **16** | English-only text | `"This is an entirely English sentence..."` | Cross-lingual transfer evaluation | Classified (HECM 82.3%, Tamil C0, Mal C1) | Yes | Normal, finite attributions |
| **17** | Multi-script code-mixed| `"Super video भाई sema mass padam padam"` | Multi-script token alignment | Classified (Finite, sum = 1.0) | Yes | Normal, finite attributions |
| **18** | Combining marks/accents| `"Hátè spëëch cömbïnïng \u0300\u0301\u0302"` | Normalize and ingest Unicode safely | Classified (Finite, sum = 1.0) | Yes | Normal, finite attributions |
| **19** | Zero-width / spaces | `"Hate\u200b\u200c\u200dSpeech\xa0\u2002\u3000Test"` | Strip invisible chars; decode clean tokens | Classified (`HateSpeech Test`) | Yes | Normal, finite attributions |
| **20** | Very long input (>128 tok)| 250 words (`"word0 word1 ... word249"`) | Enforce strictly $\le 128$ tokens ($126$ content) | Truncated to exactly 128 tokens | Yes | Boundary warning rendered |

---

## 3. Discovered Vulnerabilities & Fixes Implemented

### Bug 1: Unchecked Empty and Whitespace Inputs in Inference Engine
- **Vulnerability:** `predict_sample()` accepted empty strings and whitespace-only text, producing `visible_text = ""` and feeding empty `[CLS], [SEP]` tokens into the transformer. The model silently classified the empty sequence into an arbitrary class.
- **Fix:** Added strict validation in `app/utils/model_loader.py:predict_sample()`:
  ```python
  if not isinstance(text, str) or not text.strip():
      raise ValueError("Input text must contain non-whitespace characters.")
  if not visible_text.strip():
      raise ValueError("Input text produced no visible tokens for model evaluation.")
  ```

### Bug 2: Stale State Retention Upon Clearing Input
- **Vulnerability:** In `app/app.py`, clicking "Analyze Text" when the text area was empty or whitespace produced a warning, but left `st.session_state["last_result"]` intact. Consequently, the previous analysis results and attributions remained displayed immediately below the empty-input warning.
- **Fix:** In `app/app.py`, the empty-input branch now explicitly resets `st.session_state["last_result"]`, `st.session_state["shap_scores"]`, `st.session_state["attention_scores"]`, and `st.session_state["last_analyzed_input"]`.

### Bug 3: Ambiguity Between Modified Input and Displayed Predictions
- **Vulnerability:** When a user edited the text in the input box or selected a different preset after a previous analysis, the prediction panel continued to display the previous results without any indicator that the input had changed.
- **Fix:**
  1. Stored `last_analyzed_input` in session state.
  2. In Tab 1, added an immediate warning banner if `user_input.strip() != last_analyzed_input`:  
     `"⚠️ Input Modified: The text in the input box differs from the analyzed text below. Click [🚀 Analyze Text] to evaluate your updated text."`
  3. In `app/components/prediction.py`, explicitly rendered `Evaluated Text: <visible_text>` at the top of the prediction card.

### Bug 4: Unhandled Exceptions Escaping from SHAP & Attention Maskers
- **Vulnerability:** Neither `compute_shap_word_scores` nor `compute_attention_word_scores` were defensively guarded against runtime exceptions, out-of-bounds target classes, or non-finite outputs. Any tokenizer mismatch or SHAP masker error would crash the entire Streamlit application.
- **Fix:**
  1. In `app/components/explanations.py`, wrapped both functions in defensive `try...except` blocks that validate array dimensions, filter non-finite scores (`np.isfinite`), and return `[]` gracefully.
  2. In `app/app.py`, wrapped inference, Attention, and SHAP calls in separate `try...except` blocks with informative UI notifications (`st.warning`, `st.error`).

### Bug 5: Permissive `MODEL_SOURCE` Falling Back Silently
- **Vulnerability:** `get_model_source()` silently defaulted to `"hf"` on invalid environment variable settings (e.g., `MODEL_SOURCE=invalid`), hiding configuration errors.
- **Fix:** Added strict validation in `app/utils/model_loader.py:get_model_source()`:
  ```python
  raw = os.environ.get("MODEL_SOURCE", "hf")
  source = raw.strip().lower() if raw and raw.strip() else "hf"
  if source not in ("hf", "local"):
      raise ValueError(f"Invalid MODEL_SOURCE '{raw}'. Supported options are 'hf' or 'local'.")
  ```

---

## 4. Phase C — Model Output Integrity

Verified across all 3 language settings:
- **HECM:** Logits shape = $(1, 2)$, Class indices $\in \{0, 1\}$, Labels: `{0: "Non-Hate", 1: "Hate"}`.
- **Tamil:** Logits shape = $(1, 6)$, Class indices $\in \{0, \dots, 5\}$, Labels: `{0: "Not-offensive (label 0)", ..., 5: "Offensive_Category_5"}`.
- **Malayalam:** Logits shape = $(1, 5)$, Class indices $\in \{0, \dots, 4\}$, Labels: `{0: "Not_offensive", 1: "not-malayalam", ..., 4: "Offensive_Targeted_Insult_Group"}`.

### Language Switching Invariant Test
Tested cyclical switching: `HECM` $\to$ `Tamil` $\to$ `Malayalam` $\to$ `HECM`.
- All class mappings remained isolated and correct ($2 \to 6 \to 5 \to 2$).
- No cross-contamination of previous model predictions or state.
- All probabilities $\sum_{i} P(y_i) = 1.0 \pm 10^{-4}$ and $\forall i: P(y_i) \in [0, 1]$.

---

## 5. Phase D & E — Explainability Integrity (SHAP & Attention)

- **SHAP Token Attribution:**
  - Tested on 1-token input, short input, multilingual input, punctuation-heavy input, repeated tokens, and truncated long inputs.
  - Attributions are finite, correctly aggregated to word spans, and tied to the evaluated text.
  - Subword aggregation maps cleanly to visible words.
- **Self-Attention Extraction:**
  - Extracted from Layer 12, averaged across all 12 heads.
  - Special tokens (`[CLS]`, `[SEP]`, `[PAD]`) excluded from attribution ranking.
  - Routing weights normalized and finite.
- **Scientific Honesty Disclaimers Preserved:**
  - Attention disclaimer strictly states:  
    *“Attention did not show statistically significant behavioral relevance in the top-vs-random deletion comparison in any of the three evaluated language settings. This does not establish that attention can never be useful as an explanation.”*
  - Prediction disclaimer strictly states:  
    *“Predictions are model outputs, not ground-truth labels or definitive judgments that a text is hateful or offensive.”*

---

## 6. Phase F & G — State, Network, and Source Verification

- **State Interactions:**
  - Clearing input $\to$ State reset, old predictions eliminated.
  - Switching language $\to$ Session state cleared, checkpoint reloaded cleanly.
  - Rerunning page / Tab switching $\to$ UI remains stable, session state preserved.
  - Repeated analysis clicks $\to$ Deterministic, no memory leakage.
- **Model Source Selection:**
  - `MODEL_SOURCE=local`: Loads from `results/modeling/`, verifies 2.0 GB checkpoint footprint.
  - `MODEL_SOURCE=hf`: Connects to verified Hugging Face Hub repositories (`sohilshaik07/team6-mbert-*`).
  - Missing or invalid source: Actionable error raised with supported options (`hf`, `local`).

---

## 7. Phase H — Automated Test Suite Results

Full automated test suite executed:
```bash
python -m pytest tests/ -v
```

### Complete Test Results (40/40 PASSED)
```text
tests/test_app.py::TestStreamlitApp::test_app_modules_import PASSED                      [  2%]
tests/test_app.py::TestStreamlitApp::test_attention_scores_structure PASSED             [  5%]
tests/test_app.py::TestStreamlitApp::test_checkpoint_paths_and_safetensors_exist PASSED [  7%]
tests/test_app.py::TestStreamlitApp::test_checkpoint_storage_info PASSED                 [ 10%]
tests/test_app.py::TestStreamlitApp::test_label_mappings_counts PASSED                   [ 12%]
tests/test_app.py::TestStreamlitApp::test_prediction_output_shape_and_probability_sum PASSED [ 15%]
tests/test_app.py::TestStreamlitApp::test_sample_presets_presence PASSED                 [ 17%]
tests/test_app.py::TestStreamlitApp::test_supported_languages PASSED                     [ 20%]
tests/test_calibration.py::TestCalibration::test_brier_score_perfect PASSED             [ 22%]
tests/test_calibration.py::TestCalibration::test_brier_score_worst PASSED               [ 25%]
tests/test_calibration.py::TestCalibration::test_ece_from_bins PASSED                    [ 27%]
tests/test_calibration.py::TestCalibration::test_extract_confidences PASSED             [ 30%]
tests/test_calibration.py::TestCalibration::test_reliability_bins_structure PASSED     [ 32%]
tests/test_edge_cases.py::TestEdgeCasesAndInvariants::test_attention_empty_input_graceful PASSED [ 35%]
tests/test_edge_cases.py::TestEdgeCasesAndInvariants::test_attention_exception_handled_gracefully PASSED [ 37%]
tests/test_edge_cases.py::TestEdgeCasesAndInvariants::test_empty_and_whitespace_input_rejected PASSED [ 40%]
tests/test_edge_cases.py::TestEdgeCasesAndInvariants::test_language_switching_class_counts_and_mappings PASSED [ 42%]
tests/test_edge_cases.py::TestEdgeCasesAndInvariants::test_model_identifier_invalid_dataset_key PASSED [ 45%]
tests/test_edge_cases.py::TestEdgeCasesAndInvariants::test_model_source_parsing_and_validation PASSED [ 47%]
tests/test_edge_cases.py::TestEdgeCasesAndInvariants::test_shap_empty_input_graceful PASSED [ 50%]
tests/test_edge_cases.py::TestEdgeCasesAndInvariants::test_shap_out_of_bounds_class_handled_gracefully PASSED [ 52%]
tests/test_edge_cases.py::TestEdgeCasesAndInvariants::test_truncation_boundary_invariants PASSED [ 55%]
tests/test_loaders.py::TestDataLoaders::test_load_split_data_slice PASSED                 [ 57%]
tests/test_loaders.py::TestDataLoaders::test_raw_dataset_shapes PASSED                   [ 60%]
tests/test_loaders.py::TestDataLoaders::test_split_indices_structure PASSED             [ 62%]
tests/test_loaders.py::TestDataLoaders::test_split_integrity_zero_leakage PASSED         [ 65%]
tests/test_metrics.py::TestMetrics::test_accuracy PASSED                                 [ 67%]
tests/test_metrics.py::TestMetrics::test_compute_all_metrics PASSED                      [ 70%]
tests/test_metrics.py::TestMetrics::test_confusion_matrix PASSED                         [ 72%]
tests/test_metrics.py::TestMetrics::test_macro_and_weighted_f1 PASSED                    [ 75%]
tests/test_metrics.py::TestMetrics::test_per_class_metrics PASSED                        [ 77%]
tests/test_model_registry.py::TestModelRegistry::test_checkpoint_paths_exist PASSED     [ 80%]
tests/test_model_registry.py::TestModelRegistry::test_label_mappings PASSED             [ 82%]
tests/test_model_registry.py::TestModelRegistry::test_model_specs PASSED                 [ 85%]
tests/test_model_registry.py::TestModelRegistry::test_tokenizer_loading PASSED           [ 87%]
tests/test_paths.py::TestPaths::test_core_directories_exist PASSED                       [ 90%]
tests/test_robustness.py::TestRobustness::test_character_script_classification PASSED   [ 92%]
tests/test_robustness.py::TestRobustness::test_load_authoritative_formatting_robustness PASSED [ 95%]
tests/test_robustness.py::TestRobustness::test_load_authoritative_script_composition PASSED [ 97%]
tests/test_robustness.py::TestRobustness::test_text_script_composition PASSED           [100%]

============================= 40 passed in 9.08s ==============================
```

### Smoke Test Verification
- `scripts/smoke_test_app.py` with `MODEL_SOURCE=local`: **ALL 3 LANGUAGE PIPELINES PASS**
- `scripts/smoke_test_app.py` with `MODEL_SOURCE=hf`: **ALL 3 LANGUAGE PIPELINES PASS**

### Deterministic Parity Test Verification
```text
======================================================================
PARITY TEST SUMMARY (scripts/parity_test_local_vs_hf.py)
======================================================================
  HECM          max|diff| = 0.00e+00  [PASS]
  TAMIL         max|diff| = 0.00e+00  [PASS]
  MALAYALAM     max|diff| = 0.00e+00  [PASS]

Overall result: ALL PASS
======================================================================
```

---

## 8. Phase I — Research Integrity Verification

All authoritative historical research files and model checkpoints were verified and remain **100% byte-identical and unmodified**:

| File Path | Byte Size | SHA256 Prefix | Integrity Status |
| :--- | :---: | :---: | :---: |
| `results/final/master_experimental_results.csv` | 983 B | `c4541ace7f984a82` | **VERIFIED UNCHANGED** |
| `results/calibration/calibration_results.csv` | 340 B | `728c07fdd9b6629c` | **VERIFIED UNCHANGED** |
| `results/xai/tamil_xai_summary_final.json` | 1,495 B | `ce585ba2843ede27` | **VERIFIED UNCHANGED** |
| `results/xai/malayalam_xai_summary.json` | 1,041 B | `dc2d32618ef5f56a` | **VERIFIED UNCHANGED** |
| `results/robustness/formatting_robustness_results.csv` | 698 B | `05626d3c73782cb2` | **VERIFIED UNCHANGED** |
| `results/robustness/script_composition_per_class_metrics.csv`| 1,793 B | `2a2900117e7ecd72` | **VERIFIED UNCHANGED** |
| `results/modeling/mbert_hecm/checkpoint-824/model.safetensors` | 678.5 MB | `(binary weights)` | **VERIFIED UNCHANGED** |
| `results/modeling/mbert_tamil/checkpoint-3844/model.safetensors` | 678.5 MB | `(binary weights)` | **VERIFIED UNCHANGED** |
| `results/modeling/mbert_malayalam/checkpoint-1756/model.safetensors` | 678.5 MB | `(binary weights)` | **VERIFIED UNCHANGED** |

---

## 9. Known Limitations & Unresolved Issues

- **Known Limitations:**
  - *L1 (Architecture Scope):* Evaluates mBERT only; other Indic transformers are outside this demo's scope.
  - *L2 (SHAP Computational Bound):* Real-time KernelSHAP on CPU takes 2–15 seconds per input depending on word count. Input is strictly bounded to 126 content tokens to guarantee UI responsiveness.
  - *L3 (OutOfDistribution Syntax under Deletion):* Word deletions alter syntax, as documented in the research report.
- **Unresolved Issues:** **None.** All identified vulnerabilities, edge cases, and state anomalies have been remediated, verified with automated unit tests, and validated in end-to-end execution.
