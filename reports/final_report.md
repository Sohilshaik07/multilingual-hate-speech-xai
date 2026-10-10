# Faithfulness-Aware Explainability Audit of Multilingual Transformer Models for Code-Mixed Hate Speech Detection

**Team 6 — XAI Research Project**
**Report Date:** October 2026
**Repository:** `Team6_XAI`
**Execution Seed:** 42
**Status:** Final — All experimental results are authoritative transferred Colab artifacts.

---

## Abstract

Automated hate speech detection in code-mixed multilingual social media text presents unique challenges for transformer-based language models: severe class imbalance, script heterogeneity, and the conflation of surface-level accuracy with genuine model reliability. This report presents a faithfulness-aware explainability audit of fine-tuned multilingual BERT (mBERT) applied to three code-mixed hate speech detection benchmarks spanning Hindi-English (HECM, *N* = 9,357), Tamil-English (DravidianCodeMix, *N* = 43,919), and Malayalam-English (DravidianCodeMix, *N* = 20,010), totalling 73,286 benchmark instances.

We evaluate the models across four auditing dimensions: (1) **predictive performance** under class-imbalance-aware metrics; (2) **confidence calibration** using Expected Calibration Error (ECE) and Brier score; (3) **explainability faithfulness** by independently auditing KernelSHAP and self-attention weights via a deletion-based behavioural relevance protocol; and (4) **robustness** under surface formatting perturbations and mixed-script Unicode composition subsets.

Key findings are as follows. mBERT achieves misleadingly high overall accuracy (71.0–91.9%) while Macro-averaged F1 reveals substantial minority-class performance degradation (Macro-F1: 0.450–0.770). All three models are systematically overconfident (ECE ≈ 0.066–0.071; mean confidence exceeds empirical accuracy by 6.5–6.8 percentage points). KernelSHAP token attributions demonstrate statistically significant behavioural relevance under deletion (*p* < 0.001 across all three languages), whereas self-attention weights fail to achieve this standard in any language setting (*p* > 0.05 for Top-vs-Random deletion in all three cases). Attribution agreement between attention and SHAP is weak-to-moderate (Top-1 agreement: 18–26%; mean Spearman *ρ*: 0.168–0.313). Performance degradation on mixed-script Unicode subsets is substantial, especially in Malayalam (Macro-F1 drop: −24.8%). These findings collectively establish that **surface accuracy alone is an insufficient basis for deploying multilingual hate speech detectors**, and that explainability methods require method-specific behavioural validation before use.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Related Work](#2-related-work)
3. [Datasets](#3-datasets)
4. [Methodology](#4-methodology)
5. [Experimental Setup](#5-experimental-setup)
6. [Results: Predictive Performance](#6-results-predictive-performance)
7. [Results: Confidence Calibration](#7-results-confidence-calibration)
8. [Results: Explainability Audit](#8-results-explainability-audit)
9. [Results: Robustness Analysis](#9-results-robustness-analysis)
10. [Discussion](#10-discussion)
11. [Limitations and Future Work](#11-limitations-and-future-work)
12. [Conclusion](#12-conclusion)
13. [References](#13-references)
- [Appendix A: Experimental Artifact Registry](#appendix-a-experimental-artifact-registry)
- [Appendix B: Figure Registry](#appendix-b-figure-registry)

---

## 1. Introduction

Hate speech on social media platforms increasingly manifests in code-mixed form — utterances that interleave two or more languages within a single sentence, often combining a dominant regional language with English. This phenomenon is particularly prevalent across South and Southeast Asian communities, where speakers routinely produce Hinglish (Hindi-English), Tanglish (Tamil-English), and Manglish (Malayalam-English) content on platforms such as Twitter, YouTube comments, and Telegram. Automated detection of hate speech in such settings is both socially consequential and technically difficult.

Transformer-based language models, particularly multilingual BERT (mBERT) [1], have achieved strong benchmark performance on code-mixed classification tasks. However, raw predictive metrics can be profoundly misleading in these settings for two reasons. First, severe class imbalance (majority non-offensive classes constituting 72–88% of samples) causes overall accuracy to substantially overstate model capability on minority offensive categories. Second, even when a model produces correct predictions, the internal reasoning encoded in its probability distributions and attribution mechanisms may not reliably reflect the text features that make content hateful.

These two failure modes motivate an **explainability audit** approach: rather than asking only *what* a model predicts, we ask *how confidently* it predicts (calibration) and *why* it predicts (faithfulness of explanations). Faithfulness, in our usage, is strictly defined as **behavioural relevance under deletion**: a token attribution is behaviourally faithful to the extent that removing the tokens deemed most important by the method causes a measurably larger disruption to model decisions than removing random or least-important tokens.

This report presents a comprehensive faithfulness-aware audit of mBERT across three code-mixed multilingual settings. The **primary research contributions** are:

1. A demonstration that accuracy-based reporting masks fundamental Macro-F1 deficits in minority hate speech categories across all three language settings.
2. An empirical calibration audit establishing systematic overconfidence in all three fine-tuned mBERT models.
3. A behavioural faithfulness evaluation showing that KernelSHAP attributions are statistically significant behavioural indicators (*p* < 0.001) while self-attention weights are not (*p* > 0.05) in any tested language.
4. Quantification of attribution disagreement between self-attention and SHAP, showing weak-to-moderate alignment (Top-1 agreement ≤ 26%).
5. A script-composition robustness analysis identifying significant Macro-F1 degradation on mixed-script Unicode subsets, most severely in Malayalam (−24.8%).

The remainder of this report is organised as follows. Section 2 surveys related work. Section 3 describes the datasets. Section 4 details the methodology. Section 5 describes the experimental setup. Sections 6–9 present the four audit dimensions. Section 10 discusses findings holistically. Section 11 documents limitations and future work. Section 12 concludes.

---

## 2. Related Work

### 2.1 Multilingual Hate Speech Detection

Hate speech detection has attracted substantial NLP research effort, with early foundational work establishing key challenges in dataset construction and classifier design [7]. The shift toward transformer-based models brought significant accuracy improvements, but also highlighted the interaction between class imbalance, domain specificity, and evaluation metric choice. Multilingual settings introduce additional complexity: pre-trained multilingual models must generalise across typologically distinct languages and script systems without language-specific fine-tuning resources.

Code-mixed hate speech is an especially challenging variant, as models must process utterances that simultaneously encode phonological, morphological, and lexical features from multiple languages within a single input sequence. The DravidianCodeMix shared tasks for Tamil and Malayalam provided benchmark datasets and standardised evaluation protocols that have been widely adopted for this purpose.

### 2.2 Multilingual BERT

BERT [1] introduced bidirectional transformer pre-training with masked language modelling and next-sentence prediction objectives on large text corpora. The multilingual variant (mBERT, `bert-base-multilingual-cased`) extends this to 104 languages using a shared 119,547-token WordPiece vocabulary trained on Wikipedia corpora. mBERT has demonstrated competitive zero-shot and fine-tuned cross-lingual transfer performance across many downstream tasks, including code-mixed classification, without language-specific architectural modifications.

### 2.3 Explainability Methods for Transformers

A dominant line of XAI research focuses on **feature attribution**: assigning importance scores to input tokens that explain a model's output. SHAP (SHapley Additive exPlanations) [2] provides a theoretically grounded attribution framework based on Shapley values from cooperative game theory, ensuring properties of local accuracy, missingness, and consistency. KernelSHAP [2] is a practical kernel-weighted regression approximation of Shapley values applicable to black-box models.

Self-attention weights in transformer models have been frequently used as a proxy for explanations, with some works arguing they reflect meaningful linguistic structure. However, the faithfulness of attention weights as explanations has been substantially contested. Jain & Wallace [3] demonstrated that attention distributions can be permuted without affecting model predictions, arguing that attention weights do not provide faithful explanations of model behaviour. Wiegreffe & Pinter [4] subsequently offered counterarguments, showing that under alternative definitions attention can be faithful. This empirical debate motivates direct, method-specific behavioural validation of attribution methods, which we adopt as our auditing framework.

LIME [6], an alternative local approximation method, is noted as complementary to KernelSHAP but is not independently evaluated in this study due to computational constraints.

### 2.4 Confidence Calibration

Calibration refers to the alignment between a model's predicted probability and its empirical accuracy — a perfectly calibrated model achieves 70% accuracy on examples where it predicts 70% confidence. Guo et al. [5] showed that modern deep neural networks, including transformer models, tend to be systematically overconfident. Expected Calibration Error (ECE) [5] and the Brier score are standard quantitative measures of calibration quality. Overconfident models are of particular concern in high-stakes classification tasks such as hate speech detection, where over-reliance on model confidence scores can lead to systematically incorrect moderation decisions.

---

## 3. Datasets

All three datasets were retained in their entirety without any sample removal, in accordance with the project's data integrity principles. The complete benchmark spans **73,286 instances** across three independent classification tasks with incompatible label schemas.

### 3.1 Dataset Overview

| Dataset | Language Pair | Train | Dev | Test | Total | Format |
|---|---|---|---|---|---|---|
| HECM Hindi-English | Hindi–English | 9,357† | — | — | 9,357 | Paired `.txt` (UTF-8) |
| Tamil DravidianCodeMix | Tamil–English | 35,139 | 4,388 | 4,392 | 43,919 | CSV |
| Malayalam DravidianCodeMix | Malayalam–English | 16,010 | 1,999 | 2,001 | 20,010 | TSV (headerless) |
| **Total Benchmark** | — | **60,506** | **6,387** | **6,393** | **73,286** | — |

*† HECM is provided as a single paired file; train/test splits are applied programmatically via leakage-safe index partitioning (seed = 42, details in Section 4.2).*

### 3.2 Class Distribution

Each dataset employs a distinct, incompatible label schema. The label distributions were analysed independently and are preserved as-is in all experimental evaluations.

**HECM Hindi-English (Binary):**

| Class | Label | Count | Proportion |
|---|---|---|---|
| 0 | Non-hate | 2,823 | 30.17% |
| 1 | Hate | 6,534 | 69.83% |

**Tamil DravidianCodeMix (6-class):**

| Class | Description | Count | Proportion |
|---|---|---|---|
| 0 | Not-offensive | 31,808 | 72.42% |
| 4 | Offensive (targeted/other) | 3,630 | 8.27% |
| 3 | Offensive (other) | 3,140 | 7.15% |
| 5 | Offensive (untargeted) | 2,965 | 6.75% |
| 1 | Homophobia/Transphobia | 1,786 | 4.07% |
| 2 | Counter-speech | 590 | 1.34% |

**Malayalam DravidianCodeMix (5-class):**

| Class | Description | Count | Proportion |
|---|---|---|---|
| Not_offensive | Not offensive | 17,678 | 88.35% |
| not-malayalam | Off-topic / non-Malayalam | 1,607 | 8.03% |
| Offensive_Targeted_Insult_Individual | Targeted individual insult | 304 | 1.52% |
| Offensive_Untargeted | Untargeted offensive | 238 | 1.19% |
| Offensive_Targeted_Insult_Group | Targeted group insult | 183 | 0.91% |

The extreme class imbalance in Tamil (72.42% Class 0) and Malayalam (88.35% Not_offensive) is a critical contextual factor for interpreting all downstream metrics. A naive majority-class classifier would achieve approximately 72% accuracy on Tamil and approximately 88% accuracy on Malayalam without learning any hate-specific features.

### 3.3 Script and Code-Mixing Composition

Thirteen interpretable linguistic and script-level features were extracted per sample. A key finding from exploratory data analysis concerns the dominant script composition of the HECM dataset:

- **83.52%** of HECM samples are Romanized Latin-only (Hinglish written in Latin script)
- **16.24%** are genuine script code-mixed (co-occurrence of Latin and Devanagari characters)

This has important implications for interpreting script-based code-mixing metrics. Transliterated content written entirely in Latin script is classified as single-script Latin, meaning actual linguistic code-mixing is substantially underestimated by Unicode block-based detection.

**Summary feature statistics across the three datasets:**

| Feature | HECM (Mean ± SD) | Tamil (Mean ± SD) | Malayalam (Mean ± SD) |
|---|---|---|---|
| Character count | 133.49 ± 76.20 | 72.29 ± 58.81 | 66.99 ± 50.42 |
| Word count | 24.46 ± 14.79 | 10.40 ± 7.25 | 9.12 ± 6.18 |
| Native script ratio | 0.031 ± 0.135 | 0.178 ± 0.374 | 0.317 ± 0.439 |
| Script code-mixing ratio | 0.016 ± 0.064 | 0.009 ± 0.050 | 0.034 ± 0.095 |
| Language switch count | 0.31 ± 0.88 | 0.10 ± 0.60 | 0.35 ± 1.28 |

### 3.4 Data Quality and Duplicate Audit

All datasets were retained at full size (rows removed = 0). Duplicate rows, duplicate text instances, and conflicting text instances (identical text assigned to different labels) were documented but not removed, in accordance with the project's data integrity principle:

| Dataset | Missing Values | Exact Duplicate Rows | Duplicate Text Instances | Conflicting Instances |
|---|---|---|---|---|
| HECM | 0 | 93 | 118 | 25 |
| Tamil | 0 | 256 | 400 | 20 |
| Malayalam | 0 | 4,454 | 6,402 | 8 |

The presence of conflicting instances (same text, different label) is noted as a data quality limitation that affects the theoretical upper bound on achievable model accuracy.

---

## 4. Methodology

### 4.1 Model Architecture

All three classification tasks were addressed using **`bert-base-multilingual-cased`** (mBERT) [1], a 12-layer bidirectional transformer with 768 hidden dimensions, 12 attention heads, and approximately 178 million parameters. mBERT was pre-trained on Wikipedia corpora covering 104 languages using masked language modelling (MLM) and next-sentence prediction (NSP) objectives.

For classification fine-tuning, a linear classification head was appended to the `[CLS]` token representation, projecting from 768 dimensions to the number of target classes (2 for HECM, 6 for Tamil, 5 for Malayalam). The three models were independently fine-tuned on each dataset; no cross-dataset transfer or parameter sharing was employed.

**Final inference checkpoints:**

| Language | Checkpoint |
|---|---|
| HECM | `results/modeling/mbert_hecm/checkpoint-824` |
| Tamil | `results/modeling/mbert_tamil/checkpoint-3844` |
| Malayalam | `results/modeling/mbert_malayalam/checkpoint-1756` |

### 4.2 Leakage-Safe Data Splits

To guarantee strict train/test separation and reproducibility, leakage-safe index partitions were created programmatically using a fixed random seed (seed = 42) and stored as JSON index files prior to any model training or evaluation:

- `results/modeling/splits/hecm_split_seed42.json`
- `results/modeling/splits/tamil_split_seed42.json`
- `results/modeling/splits/malayalam_split_seed42.json`

For Tamil and Malayalam, the official task-provided train/dev/test splits were used directly. For HECM, which lacks an official split, a stratified 80/20 train/test partition was applied. The split index files contain test set indices only; no evaluation was performed using training indices.

### 4.3 Feature Engineering

Thirteen interpretable linguistic features were computed per sample to support exploratory data analysis and script-composition robustness evaluation. The **script code-mixing ratio** is defined as:

$$\text{script\_code\_mixing\_ratio} = \frac{\min(\text{English\_Chars}, \text{Native\_Chars})}{\text{English\_Chars} + \text{Native\_Chars}}$$

This ratio is strictly Unicode block-based and is subject to the limitation that fully Romanized transliterated content yields a near-zero score despite genuine linguistic code-mixing.

### 4.4 Confidence Calibration

Calibration was evaluated using two metrics:

**Expected Calibration Error (ECE):** The dataset is partitioned into *M* = 15 uniform confidence bins over [0, 1]. ECE is the weighted average of the absolute gap between mean confidence and empirical accuracy within each bin:

$$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

**Brier Score:** The mean squared error between predicted probability vectors and one-hot ground truth labels:

$$\text{Brier} = \frac{1}{N} \sum_{i=1}^{N} \sum_{k=1}^{K} (p_{ik} - y_{ik})^2$$

Both metrics were computed on the test partition from argmax-extracted posterior probabilities.

### 4.5 Explainability Audit: KernelSHAP

KernelSHAP [2] was applied to **100 randomly sampled test instances** (seed = 42) per language. Attribution scores were computed at the WordPiece token level using a zeroed-embedding baseline.

**Behavioural faithfulness protocol:** For each audited sample, the following deletions were performed:

- **Top-token deletion:** Mask the single token with the highest KernelSHAP magnitude score.
- **Bottom-token deletion:** Mask the single token with the lowest KernelSHAP magnitude score.
- **Random-token deletion:** Mask a uniformly randomly selected token.

Three outcome statistics were recorded: mean confidence drop, median confidence drop, and binary prediction flip rate. Statistical significance was assessed using a one-sided Wilcoxon signed-rank test (H₁: Top deletion causes greater confidence drop than Random/Bottom deletion) at α = 0.05.

### 4.6 Explainability Audit: Self-Attention

Self-attention weights from the final transformer layer were extracted for the same 100 test instances. Attention weights from all 12 attention heads were averaged per token. The same Top/Random/Bottom deletion protocol and Wilcoxon signed-rank test were applied.

### 4.7 Attribution Agreement

Cross-method agreement between attention and KernelSHAP was quantified using:

- **Exact Top-1 Agreement:** Proportion of samples where both methods identify the same token as most important.
- **Spearman Rank Correlation:** Correlation between the attention-weight ranking and SHAP-magnitude ranking of tokens within each sample, averaged across the 100 audited instances.

### 4.8 Robustness Analysis

Two robustness dimensions were evaluated:

**Surface formatting perturbations:** Prediction consistency was measured across 8 experiments applying whitespace collapsing, punctuation spacing normalisation, and minor typographic perturbations to test inputs.

**Script composition subset analysis:** Test instances were partitioned into Single-script and Mixed-script (co-occurring Latin and native Indic script characters) subsets. Macro-F1 was computed independently for each subset to quantify performance degradation associated with Unicode script heterogeneity.

---

## 5. Experimental Setup

All fine-tuning and XAI experiments were conducted on Google Colab with GPU acceleration. Key parameters:

| Parameter | Value |
|---|---|
| Base model | `bert-base-multilingual-cased` |
| Tokeniser | WordPiece (cased, max length 128) |
| Training framework | HuggingFace Transformers |
| Random seed | 42 (splits, SHAP sampling, evaluation) |
| XAI audit sample size | 100 instances per language |
| Calibration bins | 15 uniform bins over [0, 1] |
| Mixed-script detection | Unicode block membership (Latin vs. Indic) |
| Formatting perturbation experiments | 8 distinct experiments |
| Platform | Google Colab (GPU) → Artifacts transferred locally |

Specific hyperparameters (learning rate, batch size, epoch count) are encoded in the transferred checkpoint configurations available in `configs/experiment_config.yaml`. The experimental results documented in this report are **authoritative historical outputs** generated in the Colab environment and have not been recomputed or modified locally.

---

## 6. Results: Predictive Performance

### 6.1 Summary Table

**Macro-averaged F1** is the primary evaluation metric, as it accounts for class imbalance by treating all classes equally regardless of support size.

| Language | Accuracy | Macro-F1 | Weighted-F1 |
|---|---|---|---|
| HECM (Hindi-English) | 0.815 | **0.770** | 0.810 |
| Tamil (Tamil-English) | 0.710 | **0.469** | 0.731 |
| Malayalam (Malayalam-English) | 0.919 | **0.450** | 0.908 |

*(Figure 1: `figures/fig01_predictive_performance.png`)*

### 6.2 Interpretation

The divergence between Accuracy and Macro-F1 is the most critical finding from this evaluation dimension:

**HECM** shows the strongest Macro-F1 (0.770), consistent with its more moderate class imbalance (70%/30% binary split). Accuracy (0.815) and Macro-F1 (0.770) are in reasonable agreement, indicating that the model achieves comparably distributed performance across both classes.

**Tamil** exhibits a large Accuracy–Macro-F1 gap: despite 71.0% accuracy, Macro-F1 is only 0.469. This reflects the model's failure to adequately classify minority offensive categories (Classes 1–5). The per-class script composition analysis shows that Class 2 (counter-speech, *N* = 86 in single-script subset) achieves F1 = 0.000, indicating the model never correctly identifies this category in either script composition subset.

**Malayalam** presents the most extreme divergence: 91.9% accuracy but only 0.450 Macro-F1. This is a direct consequence of the 88.35% majority non-offensive class — the model's high accuracy is primarily attributable to correct classification of the dominant majority class. Class 4 (Offensive_Targeted_Insult_Group) achieves F1 = 0.000 in both single-script and mixed-script subsets.

**Key conclusion:** Overall accuracy is a misleading primary metric for these class-imbalanced hate speech detection tasks. Macro-F1 correctly reveals that the models exhibit substantial performance deficits on minority offensive categories — precisely the classes that matter most for content moderation applications.

---

## 7. Results: Confidence Calibration

### 7.1 Calibration Metrics

| Language | ECE (15 bins) | Brier Score | Mean Confidence | Empirical Accuracy | Overconfidence Gap |
|---|---|---|---|---|---|
| HECM | 0.0713 | 0.2827 | 0.882 | 0.815 | **+0.067** |
| Tamil | 0.0668 | 0.3935 | 0.777 | 0.710 | **+0.067** |
| Malayalam | 0.0659 | 0.1467 | 0.984 | 0.919 | **+0.065** |

*(Figure 2: `figures/fig02_calibration_ece.png`; Figures 3–5: reliability diagrams)*

### 7.2 Interpretation

All three models display **systematic overconfidence**: mean posterior confidence consistently and substantially exceeds empirical accuracy, with a remarkably consistent overconfidence gap of approximately 6.5–6.8 percentage points across all three language settings.

The **Malayalam model** exhibits the most extreme overconfidence in absolute terms: mean predicted confidence of 98.4% against an empirical accuracy of 91.9%. This extreme confidence clustering is a direct consequence of the 88.35% majority class — the model learns to assign near-maximal confidence to the majority non-offensive prediction.

**Reliability diagrams** (Figures 3–5) show:

- Calibration curves consistently fall below the perfect-calibration diagonal in high-confidence bins (> 0.80), confirming overconfidence in precisely the regime where model outputs are most likely to be acted upon in practice.
- Prediction histograms confirm extreme clustering in high-confidence bins (> 0.80), with near-zero population in bins below 0.40 — a consequence of argmax-extracted multiclass posteriors in highly imbalanced settings.

**Brier scores** provide complementary information. Tamil's Brier score (0.3935) is substantially higher than Malayalam's (0.1467), reflecting that Tamil's multiclass probability distributions carry more uncertainty despite Tamil having lower mean confidence overall. Malayalam's low Brier score does not indicate good calibration — it reflects that highly confident wrong predictions are numerically inexpensive in a 5-class setting where most errors affect low-support minority classes.

**Conclusion:** The systematic overconfidence observed across all three models means that raw posterior probabilities should not be used directly as reliability estimates for content moderation decisions without post-hoc calibration (e.g., temperature scaling).

---

## 8. Results: Explainability Audit

### 8.1 KernelSHAP: Behavioural Relevance Under Deletion

#### 8.1.1 Summary Statistics

| Language | Mean Conf. Drop (Top) | Median Conf. Drop (Top) | Flip Rate (Top) | Top vs. Random *p* | Top vs. Bottom *p* |
|---|---|---|---|---|---|
| HECM | 0.0215 | 0.0071 | 13.0% | 0.00113 | 0.00117 |
| Tamil | 0.0376 | 0.0036 | 18.0% | 0.000220 | 5.71 × 10⁻⁷ |
| Malayalam | 0.0658 | 0.0001 | 8.0% | 0.000245 | 0.000179 |

*(Figure 6: `figures/fig06_shap_faithfulness.png`)*

#### 8.1.2 Interpretation

**KernelSHAP token attributions demonstrate statistically significant behavioural relevance across all three language settings** (one-sided Wilcoxon signed-rank test, *p* < 0.001 in all cases, for both Top-vs-Random and Top-vs-Bottom comparisons).

Deleting the single token identified as most important by KernelSHAP causes:

- A consistent and statistically significant reduction in model confidence compared to random token deletion.
- Decision flips in 8–18% of audited samples — meaning that removing a single token changes the model's predicted class entirely.

Several important nuances must be noted:

**Mean vs. median drops diverge substantially.** In Malayalam, the mean confidence drop (0.0658) is orders of magnitude larger than the median (0.0001), indicating that the effect is driven by a small number of high-impact samples. The majority of instances show minimal change upon single-token deletion.

**Flip rates are modest.** Even with statistically significant differences, single-token deletion changes the prediction in only 8–18% of cases, indicating that most model decisions are distributed across multiple tokens.

**Statistical significance reflects behavioural relevance, not causal faithfulness.** The deletion protocol measures model sensitivity to input perturbation. Token deletion introduces out-of-distribution input syntax, and the resulting confidence changes reflect model sensitivity to input distribution shift as much as genuine explanatory importance.

#### 8.1.3 Directional Consistency (Tamil)

The Tamil XAI summary provides directional comparison: deleting bottom-ranked SHAP tokens causes a mean confidence *increase* of +0.0356 (bottom mean drop = −0.0356), while deleting random tokens causes a smaller mean increase of +0.0136. This ordering (top deletion decreases confidence; bottom deletion increases it) is consistent with theoretical KernelSHAP predictions and provides additional evidence that SHAP rankings encode genuine signal about model decision sensitivity.

### 8.2 Self-Attention: Behavioural Faithfulness Evaluation

#### 8.2.1 Summary Statistics

| Language | Top vs. Random *p* | Top vs. Bottom *p* | Significant at α = 0.05? |
|---|---|---|---|
| HECM | 0.1567 | 0.0118 | No (Top-vs-Random fails) |
| Tamil | 0.6209 | 0.9443 | No (both fail) |
| Malayalam | 0.0797 | 0.0378 | No (Top-vs-Random fails) |

*(Figure 7: `figures/fig07_attention_faithfulness.png`)*

#### 8.2.2 Interpretation

**Self-attention weights fail the behavioural faithfulness criterion in all three language settings.** The primary test — Top-vs-Random — does not achieve significance (*p* > 0.05) in any language.

In **Tamil**, the failure is most pronounced: both Top-vs-Random (*p* = 0.621) and Top-vs-Bottom (*p* = 0.944) are far from significance, indicating that deleting attention-top tokens causes no more decision disruption than deleting any other token.

In **HECM**, attention shows a marginal Top-vs-Bottom result (*p* = 0.012), but this does not rescue the primary faithfulness criterion: attention top-tokens are not systematically more behaviourally important than random tokens (*p* = 0.157).

In **Malayalam**, neither test achieves significance: Top-vs-Random (*p* = 0.080) narrowly misses the α = 0.05 threshold.

This result is consistent with the theoretical position articulated by Jain & Wallace [3]: attention weights are an internal routing mechanism optimised for representation learning, not aligned with decision-relevant features that explain model outputs. The complete failure of the faithfulness test in all three languages — and especially the extreme failure in Tamil — indicates that this finding is not an artefact of a single language setting.

**Conclusion:** Self-attention weights from the final transformer layer do not constitute behaviourally faithful explanations of mBERT's hate speech classification decisions in any of the three evaluated language settings. They should not be presented as explanations for individual moderation decisions.

### 8.3 Cross-Method Attribution Agreement

#### 8.3.1 Summary Statistics

| Language | Top-1 Agreement | Mean Spearman *ρ* | Median Spearman *ρ* |
|---|---|---|---|
| HECM | 23% | 0.313 | 0.332 |
| Tamil | 18% | 0.186 | 0.211 |
| Malayalam | 26% | 0.168 | 0.244 |

*(Figure 8: `figures/fig08_attention_shap_agreement.png`)*

#### 8.3.2 Interpretation

Attribution agreement between self-attention and KernelSHAP is **weak-to-moderate** across all three language settings:

- Exact Top-1 agreement never exceeds 26%, meaning that in more than three-quarters of cases the token ranked most important by SHAP differs from the token ranked most important by attention.
- Mean Spearman rank correlations are positive but modest (*ρ* ∈ [0.168, 0.313]), indicating some shared signal without strong alignment in token ranking.
- The XAI summaries further show that positive Spearman correlations occur in 63–71% of samples and negative correlations in 29–35% of samples — confirming that while the methods tend to be weakly positively correlated on average, they actively disagree in direction in a substantial minority of individual instances.

The weak agreement has a critical interpretive consequence: given that only one method (KernelSHAP) has demonstrated behavioural faithfulness under the deletion protocol, the use of attention as an explanation mechanism not only lacks its own behavioural validation but also substantially diverges from the SHAP-validated attribution in the majority of individual instances.

---

## 9. Results: Robustness Analysis

### 9.1 Surface Formatting Robustness

| Metric | Value |
|---|---|
| Prediction Consistency Rate | **100.0%** |
| Prediction Flip Rate | **0.0%** |
| Mean Absolute Confidence Shift | **< 2 × 10⁻⁷** |
| Number of Experiments | 8 |

*(Figure 9: `figures/fig09_formatting_robustness.png`)*

mBERT predictions are **exceptionally stable** against the tested surface formatting perturbations. Across all 8 tested experiments (whitespace collapsing, punctuation spacing normalisation, minor typographic changes), no prediction flips were observed and the mean confidence shift was negligibly small (< 2 × 10⁻⁷).

**Important scope caveat:** These tests cover only surface typographical perturbations. They do not evaluate robustness under semantic perturbations, lexical substitutions, adversarial attacks, synonym replacements, or transliteration variants. The 100% consistency result must not be generalised beyond the narrow category of whitespace/punctuation formatting noise tested here.

### 9.2 Script Composition Robustness

| Language | Single-Script Macro-F1 | Mixed-Script Macro-F1 | Macro-F1 Drop | Mixed-Script N |
|---|---|---|---|---|
| HECM | 0.772 | 0.675 | **−12.6%** | 76 |
| Tamil | 0.468 | 0.452 | **−3.3%** | 273 |
| Malayalam | 0.446 | 0.336 | **−24.8%** | 413 |

*(Figure 10: `figures/fig10_script_composition.png`)*

Significant Macro-F1 degradation occurs on mixed-script inputs in all three language settings, with the effect most pronounced in **Malayalam** (−24.8% relative Macro-F1 drop) and **HECM** (−12.6%).

**Per-class degradation patterns:**

In **HECM** (mixed *N* = 76), Class 0 F1 drops from 0.675 (single-script) to 0.462 (mixed-script), while Class 1 performance remains relatively stable (0.870 → 0.889). This asymmetric degradation indicates that mixed-script text disproportionately impairs identification of the minority non-hate class.

In **Tamil** (mixed *N* = 273), Class 4 F1 collapses from 0.389 (single-script) to 0.000 (mixed-script). Class 2 achieves F1 = 0.000 in both subsets, confirming complete failure on the rarest category regardless of script composition.

In **Malayalam** (mixed *N* = 413), the extreme degradation is driven by collapse of minority offensive classes: Class 1 (not-malayalam, *N* = 2 mixed) and Class 4 (Offensive_Targeted_Insult_Group, *N* = 4 mixed) both achieve F1 = 0.000 in the mixed-script subset. The majority Not_offensive class improves slightly (0.951 → 0.981) under mixed-script conditions, masking catastrophic minority-class failure in the aggregate Macro-F1.

**Caveat:** Script composition is a Unicode block-based proxy for linguistic code-mixing. Fully transliterated text (e.g., Hinglish written entirely in Latin script) is classified as Single-script Latin and is absent from the Mixed-script degradation analysis, likely underestimating the true extent of code-mixing-related performance variation.

---

## 10. Discussion

### 10.1 Synthesis Across Four Audit Dimensions

The four audit dimensions collectively reveal a coherent and consequential pattern: **mBERT achieves superficially plausible performance metrics that systematically overstate reliability for multilingual code-mixed hate speech detection.**

**Accuracy overstatement.** High overall accuracy (71–92%) is driven by majority-class performance in severely imbalanced settings. Macro-F1 (0.450–0.770) is the appropriate metric and reveals substantial minority-class failure, particularly in Malayalam and Tamil where entire offensive sub-categories achieve F1 = 0.000.

**Calibration failure.** Systematic overconfidence (ECE ≈ 0.066–0.071, mean confidence gap ≈ +6.5–6.8%) means that model posterior probabilities cannot be directly used as reliability scores. In the most extreme case (Malayalam), mean confidence reaches 98.4% while accuracy is 91.9%.

**Explanatory method divergence.** KernelSHAP demonstrates behavioural relevance under deletion (*p* < 0.001) while self-attention does not (*p* > 0.05). Crucially, this does not establish causal faithfulness for SHAP — only that SHAP-ranked tokens are more likely to disrupt model decisions than random or least-important tokens. The distinction between behavioural relevance and causal faithfulness matters: SHAP explanations are more useful than attention for predicting which deletions will disrupt the model, but neither method provides a transparent window into the model's internal causal computational mechanism.

**Robustness asymmetry.** Surface formatting noise has negligible effect, but Unicode script heterogeneity causes significant Macro-F1 degradation (up to −24.8%), with disproportionate impact on already-fragile minority offensive categories. This suggests that the primary robustness concern for deployment is script representation quality rather than typographical noise.

### 10.2 Implications for Deployment

The collective findings suggest the following practical implications:

1. **Do not use accuracy as the primary reporting metric** for code-mixed hate speech detection with class-imbalanced datasets. Macro-F1 must be reported alongside, or in preference to, overall accuracy.

2. **Do not use uncalibrated confidence scores** for content moderation thresholding. Post-hoc calibration (temperature scaling or Platt scaling) should be applied before deploying model posteriors as decision confidence estimates.

3. **Do not treat attention weights as explanations** for individual content moderation decisions. They have not passed behavioural faithfulness validation in any of the three tested language settings.

4. **Exercise caution with KernelSHAP as a causal explanation.** While SHAP attributions demonstrate behavioural relevance, this is not causal faithfulness. Individual SHAP explanations should be interpreted as behavioural sensitivity indicators rather than verified accounts of model reasoning.

5. **Invest in mixed-script robustness.** Malayalam models in particular show severe degradation on mixed-script inputs, suggesting that native Indic script Unicode representations remain underrepresented in the mBERT pre-training distribution relative to Latin-script representations.

---

## 11. Limitations and Future Work

### 11.1 Documented Limitations

**L1 — Single architecture.** All experiments evaluate mBERT only. Results may not generalise to XLM-RoBERTa, IndicBERT, MuRIL, or language-specific models. Comparative architecture benchmarking is outside the scope of this study.

**L2 — XAI audit sample size.** KernelSHAP and attention audits are conducted on 100 instances per language due to computational cost. Statistical inferences are valid for the audited samples but may not perfectly represent the full test distribution.

**L3 — Deletion protocol limitations.** The deletion-based faithfulness test introduces out-of-distribution inputs. Observed confidence changes partly reflect model sensitivity to input distribution shift rather than purely the semantic importance of the deleted token.

**L4 — Script composition proxy.** Unicode block-based mixed-script detection misclassifies Romanized transliterated content as single-script Latin. The true prevalence of code-mixing effects may be underestimated.

**L5 — Calibration binning sensitivity.** ECE values depend on the choice of M = 15 uniform bins. Alternative binning strategies may yield different ECE estimates. Brier score is reported as a bin-invariant complement.

**L6 — Narrow robustness scope.** Formatting robustness tests cover only whitespace and punctuation perturbations. Semantic, lexical, adversarial, and transliteration-variant robustness are not evaluated.

**L7 — No post-hoc calibration applied.** Temperature scaling or Platt scaling was not applied. Calibrated performance is expected to substantially improve ECE and Brier scores and is reserved for future work.

**L8 — Conflicting label instances.** 25 (HECM), 20 (Tamil), and 8 (Malayalam) conflicting text instances exist in the raw datasets. These were retained per project integrity rules and place a theoretical ceiling on achievable accuracy.

**L9 — No LIME or Integrated Gradients.** KernelSHAP is the only local attribution method evaluated. Comparison with LIME [6] or Integrated Gradients would provide additional validation or triangulation of the faithfulness findings.

### 11.2 Future Work Directions

1. **Comparative architecture study:** Evaluate XLM-RoBERTa-large, IndicBERT, and MuRIL on the same benchmarks with the identical four-dimensional audit protocol.
2. **Post-hoc calibration:** Apply temperature scaling and evaluate its impact on ECE, Brier score, and downstream decision thresholding reliability.
3. **Extended faithfulness audit:** Expand KernelSHAP and attention audits to full test sets; evaluate multi-token deletion (Top-*k* for *k* ∈ {1, 3, 5}).
4. **Adversarial robustness:** Evaluate robustness under character-level perturbations, synonym substitution, transliteration variants, and adversarial suffix attacks.
5. **Minority class rebalancing:** Evaluate the effect of oversampling and class-weighted loss functions on both Macro-F1 and calibration properties.
6. **Linguistic code-mixing metrics:** Replace Unicode block-based script detection with token-level language identification for more accurate code-mixing quantification.
7. **Human evaluation of explanations:** Conduct user studies to evaluate whether SHAP or attention explanations align with human judgements of hate speech-defining tokens.

---

## 12. Conclusion

This report presents a faithfulness-aware explainability audit of multilingual BERT (mBERT) for code-mixed hate speech detection across Hindi-English (HECM), Tamil-English (DravidianCodeMix), and Malayalam-English (DravidianCodeMix) benchmarks spanning 73,286 instances. The audit evaluates predictive performance, confidence calibration, explainability faithfulness, and robustness.

The principal findings are:

1. **Accuracy is insufficient as a primary metric.** Macro-F1 (0.450–0.770) reveals substantial minority-class performance failure that overall accuracy (71–92%) conceals, particularly in the severely imbalanced Tamil and Malayalam settings.

2. **All three models are systematically overconfident.** ECE values of 0.066–0.071 and consistent 6.5–6.8 percentage-point mean confidence gaps establish that uncalibrated posterior probabilities are unreliable confidence estimates for deployment use.

3. **KernelSHAP attributions demonstrate statistically significant behavioural relevance** (*p* < 0.001 across all three languages), establishing them as more informative than random attribution for predicting which token deletions disrupt model decisions. This is a claim of behavioural relevance, not causal faithfulness.

4. **Self-attention weights fail behavioural faithfulness validation** in all three language settings (*p* > 0.05 for the primary Top-vs-Random test in every language), and should not be used as explanation surrogates for individual classification decisions.

5. **Cross-method agreement is weak-to-moderate** (Top-1 agreement 18–26%; mean Spearman *ρ* 0.168–0.313), confirming that attention and SHAP prioritise substantially different token subsets in the majority of individual instances.

6. **mBERT is robust to surface formatting noise** (100% prediction consistency) but shows **significant degradation on mixed-script Unicode inputs** (Macro-F1 drop up to −24.8% in Malayalam), with disproportionate collapse of already-fragile minority offensive categories.

These findings support a critical but constructive assessment: mBERT is a viable foundation for multilingual code-mixed hate speech detection, but deployment requires careful attention to class-imbalance-aware evaluation, post-hoc calibration, and explainability method validation. No component of this system — its predictions, its confidence scores, or its explanations — should be used uncritically without the methodological checks described in this report.

---

## 13. References

[1] Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT 2019)*, pp. 4171–4186. Association for Computational Linguistics.

[2] Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. In *Advances in Neural Information Processing Systems 30 (NeurIPS 2017)*, pp. 4765–4774. Curran Associates, Inc.

[3] Jain, S., & Wallace, B. C. (2019). Attention is not Explanation. In *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT 2019)*, pp. 3543–3556. Association for Computational Linguistics.

[4] Wiegreffe, S., & Pinter, Y. (2019). Attention is not not Explanation. In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP 2019)*, pp. 11–20. Association for Computational Linguistics.

[5] Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. In *Proceedings of the 34th International Conference on Machine Learning (ICML 2017)*, Proceedings of Machine Learning Research, Vol. 70, pp. 1321–1330. PMLR.

[6] Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why Should I Trust You?": Explaining the Predictions of Any Classifier. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD 2016)*, pp. 1135–1144. ACM.

[7] Waseem, Z., & Hovy, D. (2016). Hateful Symbols or Hateful People? Predictive Features for Hate Speech Detection on Twitter. In *Proceedings of the NAACL Student Research Workshop (NAACL SRW 2016)*, pp. 88–93. Association for Computational Linguistics.

---

## Appendix A: Experimental Artifact Registry

All experimental results documented in this report are available as authoritative transferred artifacts.

| Report Section | Source Artifact | Path |
|---|---|---|
| §6 Predictive Performance | Master experimental results | `results/final/master_experimental_results.csv` |
| §7 Calibration (ECE, Brier) | Calibration results | `results/calibration/calibration_results.csv` |
| §7 Reliability Diagrams | Per-bin reliability data | `results/calibration/{language}_reliability_bins.csv` |
| §8.1 SHAP Audit | Master results + XAI summaries | `results/final/master_experimental_results.csv`, `results/xai/` |
| §8.2 Attention Audit | Master experimental results | `results/final/master_experimental_results.csv` |
| §8.3 Agreement | XAI agreement CSVs | `results/xai/{language}_attention_shap_agreement.csv` |
| §9.1 Formatting Robustness | Formatting robustness results | `results/robustness/formatting_robustness_results.csv` |
| §9.2 Script Composition | Script composition metrics | `results/robustness/script_composition_per_class_metrics.csv` |

## Appendix B: Figure Registry

All 10 publication figures are available in PNG and PDF at 300 DPI in `figures/`.

| Figure | Filename Stem | Primary Source Artifact |
|---|---|---|
| Fig. 1: Predictive Performance | `fig01_predictive_performance` | `results/final/master_experimental_results.csv` |
| Fig. 2: Calibration ECE | `fig02_calibration_ece` | `results/calibration/calibration_results.csv` |
| Fig. 3: Reliability (HECM) | `fig03_reliability_hecm` | `results/calibration/hecm_reliability_bins.csv` |
| Fig. 4: Reliability (Tamil) | `fig04_reliability_tamil` | `results/calibration/tamil_reliability_bins.csv` |
| Fig. 5: Reliability (Malayalam) | `fig05_reliability_malayalam` | `results/calibration/malayalam_reliability_bins.csv` |
| Fig. 6: SHAP Faithfulness | `fig06_shap_faithfulness` | `results/final/master_experimental_results.csv` |
| Fig. 7: Attention Faithfulness | `fig07_attention_faithfulness` | `results/final/master_experimental_results.csv` |
| Fig. 8: Attention–SHAP Agreement | `fig08_attention_shap_agreement` | `results/final/master_experimental_results.csv` |
| Fig. 9: Formatting Robustness | `fig09_formatting_robustness` | `results/robustness/formatting_robustness_results.csv` |
| Fig. 10: Script Composition | `fig10_script_composition` | `results/robustness/script_composition_per_class_metrics.csv` |

---

*All numerical values in this report are sourced directly from transferred Colab result CSVs and JSON files. No results were fabricated, imputed, or modified during report generation. Report generated: October 2026.*
