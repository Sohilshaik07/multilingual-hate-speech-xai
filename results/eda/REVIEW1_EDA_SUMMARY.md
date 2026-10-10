# Review-1 EDA & Preprocessing Summary Report

**Project Title**: Faithfulness-Aware Explainability Audit of Multilingual Transformer Models for Code-Mixed Hate Speech Detection  
**Execution Seed**: 42  
**Data Integrity Status**: 100% Raw Data Preserved (0 Rows Removed)

---

## Technical Methodology & Data Lifecycle Pipeline

To guarantee 100% academic integrity and transparency, the pipeline clearly delineates four distinct stages:

```
[1. Raw Dataset Storage] ---> [2. Preprocessing Transformations] ---> [3. Feature Engineering] ---> [4. Visualization Formatting]
(Original Text/CSV Files)     (NFC & Whitespace Normalization)       (13 Linguistic Metrics)        (Matplotlib/Seaborn Plots)
```

1. **Raw Dataset Statistics**: Read directly from immutable source files (`Hindi/HECM.txt`, `Hindi/HECM_Class.txt`, `TAMIL/*.csv`, `MALAYALAM/*.csv`).
2. **Preprocessing Transformations**: Non-destructive normalization (Unicode NFC, whitespace collapsing, URL/user-mention tokenization) without removing any sample rows or destroying linguistic information.
3. **Derived Feature Calculations**: Exact per-sample metrics calculated programmatically across all 73,286 benchmark rows.
4. **Visualization-Only Formatting**: Plot styling, tick labeling, and dynamic unclipped y/x axis limits designed purely for visual clarity without modifying underlying numbers.

---

## 1. Problem Identification & Dataset Selection

This research focuses on evaluating and auditing hate speech detection in code-mixed social media text across three major Indian language pairs:
1. **Hindi-English** (HECM Dataset — 9,357 paired text-label instances)
2. **Tamil-English** (DravidianCodeMix Dataset — 43,919 train/dev/test instances)
3. **Malayalam-English** (DravidianCodeMix Dataset — 20,010 train/dev/test instances)

### Dataset Overview
| Dataset | Language Pair | Train | Dev | Test | Total Rows | Primary Format |
|---|---|---|---|---|---|---|
| HECM Hindi-English | Hindi-English | 9,357 | - | - | 9,357 | Paired Text & Class Files (`.txt`, UTF-8) |
| Tamil-English | Tamil-English | 35,139 | 4,388 | 4,392 | 43,919 | CSV (Comma-separated) |
| Malayalam-English | Malayalam-English | 16,010 | 1,999 | 2,001 | 20,010 | TSV (Tab-separated, no headers) |
| **Total Benchmark** | - | **60,506** | **6,387** | **6,393** | **73,286** | - |

---

## 2. HECM Script & Social Media Composition

In the HECM Hindi dataset (`datasets/Hindi/HECM.txt`, `datasets/Hindi/HECM_Class.txt`):
- **Total Rows**: 9,357
- **Latin Script Presence (`A-Za-z`)**: 9,335 (99.76%)
- **Devanagari Script Presence (`U+0900–U+097F`)**: 1,542 (16.48%)
- **Script Code-Mixed (Both Scripts)**: 1,520 (16.24%)
- **Romanized Latin Only (Hinglish)**: 7,815 (83.52%)

**Key Observation**: Over 83% of the HECM dataset is Romanized Hindi (Hinglish written in Latin script). This provides a valuable benchmark for evaluating transformer robustness on Romanized transliterated hate speech, while highlighting the importance of acknowledging the limitations of purely script-based code-mixing metrics.

---

## 3. Data Quality & Detailed Duplicate Audit

In strict compliance with research integrity requirements, **NO samples were deleted or filtered out**. All duplicate rows and text instances are retained and documented transparently.

| Dataset | Missing Values | Exact Duplicate Rows | Duplicate Text Instances | Conflicting Text Instances | Rows Removed | Final Rows |
|---|---|---|---|---|---|---|
| HECM Hindi-English | 0 | 93 | 118 | 25 | 0 | 9,357 |
| Tamil-English | 0 | 256 | 400 | 20 | 0 | 43,919 |
| Malayalam-English | 0 | 4,454 | 6,402 | 8 | 0 | 20,010 |

*Note: Conflicting text instances refer to exact duplicate text strings that are assigned to different ground-truth label classes in the raw dataset.*

---

## 4. Class Distribution Analysis (Independent Systems)

Each dataset utilizes a distinct, incompatible label schema. The label distributions are maintained independently:

### 1. HECM Hindi-English (Binary Class Labels '0' and '1')
- **Class 0**: 2,823 samples (30.17%)
- **Class 1**: 6,534 samples (69.83%)
- *Status*: Discrete binary distribution with approximately 70% Class 1 and 30% Class 0. Labels are preserved strictly as raw strings `'0'` and `'1'`.

### 2. Tamil-English (Numeric Labels 0 to 5)
- **Class 0** (Not-offensive): 31,808 samples (72.42%)
- **Class 4**: 3,630 samples (8.27%)
- **Class 3**: 3,140 samples (7.15%)
- **Class 5**: 2,965 samples (6.75%)
- **Class 1**: 1,786 samples (4.07%)
- **Class 2**: 590 samples (1.34%)
- *Status*: Severe class imbalance toward Class 0 (Not-offensive).

### 3. Malayalam-English (Categorical Labels)
- **Not_offensive**: 17,678 samples (88.35%)
- **not-malayalam**: 1,607 samples (8.03%)
- **Offensive_Targeted_Insult_Individual**: 304 samples (1.52%)
- **Offensive_Untargetede**: 238 samples (1.19%)
- **Offensive_Targeted_Insult_Group**: 183 samples (0.91%)
- *Status*: Extreme class imbalance with 88.35% majority non-offensive samples.

---

## 5. Feature Engineering Summary

13 interpretable linguistic and script-level code-mixing features were extracted for each sample:
1. `character_count`
2. `word_count`
3. `sentence_count`
4. `average_word_length`
5. `digit_count`
6. `punctuation_count`
7. `uppercase_ratio`
8. `english_character_count`
9. `native_script_character_count`
10. `english_ratio`
11. `native_script_ratio`
12. `language_switch_count`
13. `script_code_mixing_ratio`

### Feature Definition & Limitation
The **`script_code_mixing_ratio`** is defined as:
$$\text{script\_code\_mixing\_ratio} = \frac{\min(\text{English\_Chars}, \text{Native\_Chars})}{\text{English\_Chars} + \text{Native\_Chars}}$$

> [!WARNING]
> **Explicit Documented Limitation**:
> This metric is strictly **script-based** (detecting co-occurrence of Latin characters `A-Za-z` and native Indic script characters). Transliterated regional language text (e.g., Hinglish, Tanglish, Manglish written in Latin script) uses Latin characters and will be detected as English script. Consequently, transliterated code-mixed samples will yield a `script_code_mixing_ratio` near 0.0, underestimating actual linguistic code-mixing.

---

## 6. Summary Statistics

| Metric / Feature | Hindi-English (Mean ± Std) | Tamil-English (Mean ± Std) | Malayalam-English (Mean ± Std) |
|---|---|---|---|
| Character Count | 133.49 ± 76.20 | 72.29 ± 58.81 | 66.99 ± 50.42 |
| Word Count | 24.46 ± 14.79 | 10.40 ± 7.25 | 9.12 ± 6.18 |
| Native Script Ratio | 0.0310 ± 0.1345 | 0.1778 ± 0.3743 | 0.3165 ± 0.4389 |
| Script-Based Code-Mixing Ratio | 0.0163 ± 0.0638 | 0.0086 ± 0.0497 | 0.0339 ± 0.0949 |
| Language Switch Count | 0.31 ± 0.88 | 0.10 ± 0.60 | 0.35 ± 1.28 |

---

## 7. Future Work Roadmap (Review-2 and Beyond)

As specified in the Review-1 scope, model implementation and explainability audits are reserved for future project reviews:
1. Fine-tuning multilingual transformer baselines (mBERT, XLM-RoBERTa, IndicBERT).
2. Implementing XAI explanation attribution methods (SHAP, LIME, Integrated Gradients).
3. Faithfulness evaluation metrics (Comprehensiveness and Sufficiency).
4. Controlled perturbation experiments on code-mixed and script-switched tokens.
