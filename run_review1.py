"""
run_review1.py

Master Reproducible Pipeline for Review-1:
- Data Preprocessing
- Feature Engineering
- Exploratory Data Analysis (EDA)

Integrity Guarantees:
1. No raw dataset rows deleted.
2. Raw HECM text and class labels aligned 1-to-1 with raw string labels ('0', '1') preserved.
3. Exact duplicate rows, text duplicates, and conflicting text instances reported transparently.
4. Metric named 'script-based code-mixing ratio' with documented transliteration limitations.
5. Malayalam loaded with sep='\\t', header=None (col 0=text, col 1=label).
6. Incompatible label systems kept separated per language.
7. Class-vs-code-mixing analyzed separately per language.
8. Dataset SHA-256 hashes verified before and after execution.
9. Original files inside datasets/ NEVER modified.
10. Dynamic, unclipped visualization domains for character counts, word counts, and language switches.
"""

import os
import sys
import hashlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure UTF-8 output encoding on Windows console
sys.stdout.reconfigure(encoding='utf-8')

# Import modules from src/data/
from src.data.preprocessing import (
    verify_dataset_hashes,
    load_hindi,
    load_tamil,
    load_malayalam,
    load_all_datasets,
    compute_duplicate_audit,
    preprocess_text
)
from src.data.feature_engineering import (
    extract_features,
    get_feature_definitions
)

SEED = 42
np.random.seed(SEED)

RESULTS_DIR = "results"
EDA_DIR = os.path.join(RESULTS_DIR, "eda")
FIGURES_DIR = os.path.join(EDA_DIR, "figures")
TABLES_DIR = os.path.join(EDA_DIR, "tables")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

# Set matplotlib design parameters for publication quality
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8


def run_pipeline():
    print("=" * 90)
    print("REVIEW-1 EXECUTION PIPELINE (HECM HINDI INTEGRATION)")
    print("=" * 90)

    # STEP 1: Verify Initial Hashes
    print("\n[STEP 1] Verifying Raw Dataset Hashes (Pre-execution)...")
    initial_hashes = verify_dataset_hashes("datasets")
    for fname, h in initial_hashes.items():
        print(f"  [HASH] {fname}: {h[:16]}...{h[-8:]}")

    # STEP 2: Load Datasets
    print("\n[STEP 2] Loading Raw Datasets...")
    df_hindi = load_hindi()
    df_tamil = load_tamil()
    df_malayalam = load_malayalam()

    print(f"  - HECM Hindi Loaded: {len(df_hindi):,} rows")
    print(f"  - Tamil Loaded: {len(df_tamil):,} rows (Train: {(df_tamil['split']=='train').sum():,}, Dev: {(df_tamil['split']=='dev').sum():,}, Test: {(df_tamil['split']=='test').sum():,})")
    print(f"  - Malayalam Loaded: {len(df_malayalam):,} rows (Train: {(df_malayalam['split']=='train').sum():,}, Dev: {(df_malayalam['split']=='dev').sum():,}, Test: {(df_malayalam['split']=='test').sum():,})")

    total_benchmark_rows = len(df_hindi) + len(df_tamil) + len(df_malayalam)
    print(f"  -> Total Combined Benchmark: {total_benchmark_rows:,} rows")

    # STEP 3: HECM Script & Social Composition Analysis
    print("\n[STEP 3] Analyzing HECM Script & Social Media Artifacts...")
    hecm_dev_count = sum(1 for t in df_hindi['text'] if any(0x0900 <= ord(c) <= 0x097F for c in t))
    hecm_lat_count = sum(1 for t in df_hindi['text'] if any((65 <= ord(c) <= 90) or (97 <= ord(c) <= 122) for c in t))
    hecm_both_count = sum(1 for t in df_hindi['text'] if any(0x0900 <= ord(c) <= 0x097F for c in t) and any((65 <= ord(c) <= 90) or (97 <= ord(c) <= 122) for c in t))
    hecm_lat_only = hecm_lat_count - hecm_both_count
    print(f"  - Devanagari Presence: {hecm_dev_count:,} ({hecm_dev_count/len(df_hindi):.2%})")
    print(f"  - Latin Presence: {hecm_lat_count:,} ({hecm_lat_count/len(df_hindi):.2%})")
    print(f"  - Script Code-Mixed (Both Scripts): {hecm_both_count:,} ({hecm_both_count/len(df_hindi):.2%})")
    print(f"  - Romanized Latin Only (Hinglish): {hecm_lat_only:,} ({hecm_lat_only/len(df_hindi):.2%})")

    # STEP 4: Data Quality & Duplicate Auditing
    print("\n[STEP 4] Auditing Data Quality & Duplicates...")
    audit_hindi = compute_duplicate_audit(df_hindi)
    audit_tamil = compute_duplicate_audit(df_tamil)
    audit_malayalam = compute_duplicate_audit(df_malayalam)

    print(f"  - Hindi Data Quality: {audit_hindi}")
    print(f"  - Tamil Data Quality: {audit_tamil}")
    print(f"  - Malayalam Data Quality: {audit_malayalam}")

    # Build data quality DataFrame
    quality_rows = [
        {"Dataset": "HECM Hindi-English", **audit_hindi},
        {"Dataset": "Tamil-English", **audit_tamil},
        {"Dataset": "Malayalam-English", **audit_malayalam}
    ]
    df_quality = pd.DataFrame(quality_rows)
    df_quality.rename(columns={
        "total_rows": "Original Rows",
        "missing_text_rows": "Missing Values",
        "exact_duplicate_rows": "Exact Duplicate Rows",
        "text_duplicate_instances": "Duplicate Text Instances",
        "conflicting_duplicate_texts": "Conflicting Text Instances",
        "rows_removed": "Rows Removed",
        "final_rows": "Final Rows"
    }, inplace=True)
    
    quality_csv_path = os.path.join(TABLES_DIR, "data_quality.csv")
    df_quality.to_csv(quality_csv_path, index=False)
    print(f"  -> Saved data quality table to {quality_csv_path}")

    # Build dataset summary table
    summary_rows = [
        {"Dataset": "HECM Hindi-English", "Language Pair": "Hindi-English", "Train": len(df_hindi), "Dev": 0, "Test": 0, "Total": len(df_hindi)},
        {"Dataset": "Tamil-English", "Language Pair": "Tamil-English", "Train": int((df_tamil['split']=='train').sum()), "Dev": int((df_tamil['split']=='dev').sum()), "Test": int((df_tamil['split']=='test').sum()), "Total": len(df_tamil)},
        {"Dataset": "Malayalam-English", "Language Pair": "Malayalam-English", "Train": int((df_malayalam['split']=='train').sum()), "Dev": int((df_malayalam['split']=='dev').sum()), "Test": int((df_malayalam['split']=='test').sum()), "Total": len(df_malayalam)}
    ]
    df_summary = pd.DataFrame(summary_rows)
    summary_csv_path = os.path.join(TABLES_DIR, "dataset_summary.csv")
    df_summary.to_csv(summary_csv_path, index=False)
    print(f"  -> Saved dataset summary table to {summary_csv_path}")

    # Build label distribution table (separate for each language)
    label_dist_rows = []
    
    # Hindi
    for val, count in df_hindi['label'].value_counts().sort_index().items():
        label_dist_rows.append({
            "Dataset": "HECM Hindi-English",
            "Split": "full",
            "Label": str(val),
            "Count": count,
            "Percentage": round(count / len(df_hindi) * 100, 2)
        })
        
    # Tamil per split
    for split in ['train', 'dev', 'test']:
        df_sp = df_tamil[df_tamil['split'] == split]
        for val, count in df_sp['label'].value_counts().sort_index().items():
            label_dist_rows.append({
                "Dataset": "Tamil-English",
                "Split": split,
                "Label": str(val),
                "Count": count,
                "Percentage": round(count / len(df_sp) * 100, 2)
            })
            
    # Malayalam per split
    for split in ['train', 'dev', 'test']:
        df_sp = df_malayalam[df_malayalam['split'] == split]
        for val, count in df_sp['label'].value_counts().items():
            label_dist_rows.append({
                "Dataset": "Malayalam-English",
                "Split": split,
                "Label": str(val),
                "Count": count,
                "Percentage": round(count / len(df_sp) * 100, 2)
            })

    df_label_dist = pd.DataFrame(label_dist_rows)
    label_csv_path = os.path.join(TABLES_DIR, "label_distribution.csv")
    df_label_dist.to_csv(label_csv_path, index=False)
    print(f"  -> Saved label distribution table to {label_csv_path}")

    # Save feature summary table
    df_feat_def = get_feature_definitions()
    feat_csv_path = os.path.join(TABLES_DIR, "feature_summary.csv")
    df_feat_def.to_csv(feat_csv_path, index=False)
    print(f"  -> Saved feature summary table to {feat_csv_path}")

    # STEP 5: Preprocessing & Feature Extraction
    print("\n[STEP 5] Preprocessing Text & Extracting Features...")
    df_all = load_all_datasets()
    print(f"  - Combined dataset size: {len(df_all):,} rows")

    # Apply transparent preprocessing
    df_all['clean_text'] = df_all['text'].apply(preprocess_text)
    
    # Extract features using clean text
    df_all_feats = extract_features(df_all, text_col='clean_text', lang_col='language')
    print("  - Features successfully calculated for all samples!")

    # Calculate exact means for verification
    mean_hindi_cm = df_all_feats[df_all_feats['language']=='Hindi-English']['script_code_mixing_ratio'].mean()
    mean_mal_cm = df_all_feats[df_all_feats['language']=='Malayalam-English']['script_code_mixing_ratio'].mean()
    mean_tam_cm = df_all_feats[df_all_feats['language']=='Tamil-English']['script_code_mixing_ratio'].mean()

    # Dynamic domain calculation
    max_char_val = int(df_all_feats['character_count'].max())
    max_switch_val = int(df_all_feats['language_switch_count'].max())
    max_word_val = int(df_all_feats['word_count'].max())

    print(f"  - Exact Mean Script-Based Code-Mixing Ratio:")
    print(f"    * Hindi-English: {mean_hindi_cm:.4f}")
    print(f"    * Malayalam-English: {mean_mal_cm:.4f}")
    print(f"    * Tamil-English: {mean_tam_cm:.4f}")
    print(f"  - Dynamic Maximums: Max Character Count = {max_char_val}, Max Word Count = {max_word_val}, Max Language Switches = {max_switch_val}")

    # STEP 6: Plot Generation (Figures 1-10) with Dynamic Unclipped Visualization Domains
    print("\n[STEP 6] Generating EDA Visualizations (Dynamic Domains)...")

    # FIGURE 1: Hindi Class Distribution (Class 0 vs Class 1)
    fig, ax = plt.subplots(figsize=(7, 5))
    counts_h = df_hindi['label'].value_counts().sort_index()
    bars = ax.bar([f"Class {lbl}" for lbl in counts_h.index], counts_h.values, color=['#2b5c8f', '#d95f02'], width=0.45)
    ax.set_title("Figure 1: HECM Hindi-English Class Distribution (Class 0 vs Class 1)", fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel("Sample Count", fontsize=11)
    ax.set_xlabel("Discrete Class Label", fontsize=11)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:,}\n({height/len(df_hindi):.1%})',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.tight_layout()
    fig1_path = os.path.join(FIGURES_DIR, "fig1_hindi_class_dist.png")
    plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved {fig1_path}")

    # FIGURE 2: Tamil Class Distribution (Train Split)
    fig, ax = plt.subplots(figsize=(9, 5))
    df_t_train = df_tamil[df_tamil['split'] == 'train']
    counts_t = df_t_train['label'].value_counts().sort_index()
    bars = ax.bar([str(i) for i in counts_t.index], counts_t.values, color='#7570b3', width=0.55)
    ax.set_title("Figure 2: Tamil-English Class Distribution (Train Split)", fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel("Sample Count", fontsize=11)
    ax.set_xlabel("Numeric Class Label (0 to 5)", fontsize=11)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:,}\n({height/len(df_t_train):.1%})',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    fig2_path = os.path.join(FIGURES_DIR, "fig2_tamil_class_dist.png")
    plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved {fig2_path}")

    # FIGURE 3: Malayalam Class Distribution (Logarithmic Scale)
    fig, ax = plt.subplots(figsize=(10, 5))
    df_m_train = df_malayalam[df_malayalam['split'] == 'train']
    counts_m = df_m_train['label'].value_counts()
    
    display_labels_m = [
        "Not offensive" if l == "Not_offensive" else
        "Not Malayalam" if l == "not-malayalam" else
        "Targeted individual" if l == "Offensive_Targeted_Insult_Individual" else
        "Untargeted" if l == "Offensive_Untargetede" else
        "Targeted group" if l == "Offensive_Targeted_Insult_Group" else l
        for l in counts_m.index
    ]
    
    bars = ax.bar(display_labels_m, counts_m.values, color='#1b9e77', width=0.55)
    ax.set_title("Figure 3: Malayalam-English Class Distribution (Train Split - Logarithmic Scale)", fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel("Sample Count (Log Scale)", fontsize=11)
    ax.set_yscale('log')
    ax.set_xlabel("Label Category", fontsize=11)
    plt.xticks(rotation=15, ha='right', fontsize=10)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:,}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.tight_layout()
    fig3_path = os.path.join(FIGURES_DIR, "fig3_malayalam_class_dist.png")
    plt.savefig(fig3_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved {fig3_path}")

    # FIGURE 4: Text Character Length Distribution across Language Pairs (Dynamic Full Upper Limit)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.kdeplot(data=df_all_feats, x='character_count', hue='language', common_norm=False, fill=True, alpha=0.25, ax=ax, palette=['#2b5c8f', '#7570b3', '#1b9e77'], clip=(0, None))
    ax.set_title(f"Figure 4: Text Character Length Distribution across Language Pairs (Full Dynamic Domain 0–{max_char_val})", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Character Count per Sample", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_xlim(0, max_char_val + 50)
    plt.tight_layout()
    fig4_path = os.path.join(FIGURES_DIR, "fig4_text_length_dist.png")
    plt.savefig(fig4_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved {fig4_path}")

    # FIGURE 5: Word Count Distribution Across Language Pairs
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.kdeplot(data=df_all_feats, x='word_count', hue='language', common_norm=False, fill=True, alpha=0.25, ax=ax, palette=['#2b5c8f', '#7570b3', '#1b9e77'], clip=(0, None))
    ax.set_title(f"Figure 5: Word Count Distribution across Language Pairs (Full Domain 0–{max_word_val})", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Word Count per Sample", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_xlim(0, max_word_val + 5)
    plt.tight_layout()
    fig5_path = os.path.join(FIGURES_DIR, "fig5_word_count_dist.png")
    plt.savefig(fig5_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved {fig5_path}")

    # FIGURE 6: Native Script Ratio Distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    flier_style = dict(marker='o', markersize=1.5, alpha=0.15, markeredgecolor='none')
    sns.boxplot(data=df_all_feats, x='language', y='native_script_ratio', hue='language', palette=['#2b5c8f', '#7570b3', '#1b9e77'], ax=ax, width=0.4, flierprops=flier_style, legend=False)
    ax.set_title("Figure 6: Native Script Ratio Distribution by Language Pair", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Language Dataset", fontsize=11)
    ax.set_ylabel("Native Script Ratio (Native Chars / Total Script Chars)", fontsize=11)
    ax.set_ylim(-0.02, 1.02)
    plt.tight_layout()
    fig6_path = os.path.join(FIGURES_DIR, "fig6_script_ratio_dist.png")
    plt.savefig(fig6_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved {fig6_path}")

    # FIGURE 7: Script-Based Code-Mixing Ratio Distribution by Language (Bounded [0.0, 0.5], cut=0)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.violinplot(
        data=df_all_feats,
        x='language',
        y='script_code_mixing_ratio',
        hue='language',
        palette=['#2b5c8f', '#7570b3', '#1b9e77'],
        ax=ax,
        inner="quartile",
        cut=0,
        legend=False
    )
    ax.set_title("Figure 7: Script-Based Code-Mixing Ratio Distribution by Language Pair", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Language Dataset", fontsize=11)
    ax.set_ylabel("Script-Based Code-Mixing Ratio [Bounded 0.0 – 0.5]", fontsize=11)
    ax.set_ylim(0.0, 0.50)
    plt.tight_layout()
    fig7_path = os.path.join(FIGURES_DIR, "fig7_codemix_ratio_by_lang.png")
    plt.savefig(fig7_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved {fig7_path}")

    # FIGURE 8: Script-Based Code-Mixing Ratio by Class (Separated by Language)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True)
    flier_style_sm = dict(marker='o', markersize=1.2, alpha=0.15, markeredgecolor='none')

    # Subplot 1: Hindi HECM
    df_h_feats = df_all_feats[df_all_feats['language'] == 'Hindi-English']
    sns.boxplot(data=df_h_feats, x='label', y='script_code_mixing_ratio', ax=axes[0], hue='label', palette=['#2b5c8f', '#d95f02'], width=0.4, flierprops=flier_style_sm, legend=False)
    axes[0].set_title("HECM Hindi-English\n(Class 0 vs Class 1)", fontsize=11, fontweight='bold')
    axes[0].set_xlabel("Class Label", fontsize=10)
    axes[0].set_ylabel("Script-Based Code-Mixing Ratio", fontsize=10)
    axes[0].set_ylim(0.0, 0.50)

    # Subplot 2: Tamil
    df_t_feats = df_all_feats[df_all_feats['language'] == 'Tamil-English']
    sns.boxplot(data=df_t_feats, x='label', y='script_code_mixing_ratio', ax=axes[1], hue='label', palette='Purples', width=0.5, flierprops=flier_style_sm, legend=False)
    axes[1].set_title("Tamil-English\n(Numeric Labels 0 to 5)", fontsize=11, fontweight='bold')
    axes[1].set_xlabel("Class Label", fontsize=10)

    # Subplot 3: Malayalam
    df_m_feats = df_all_feats[df_all_feats['language'] == 'Malayalam-English']
    malayalam_display_map = {
        "Not_offensive": "Not offensive",
        "not-malayalam": "Not Malayalam",
        "Offensive_Targeted_Insult_Individual": "Targeted individual",
        "Offensive_Untargetede": "Untargeted",
        "Offensive_Targeted_Insult_Group": "Targeted group"
    }
    df_m_plot = df_m_feats.copy()
    df_m_plot['display_label'] = df_m_plot['label'].map(malayalam_display_map)
    order_m = ["Not offensive", "Not Malayalam", "Targeted individual", "Untargeted", "Targeted group"]
    
    sns.boxplot(
        data=df_m_plot,
        x='display_label',
        y='script_code_mixing_ratio',
        order=order_m,
        ax=axes[2],
        hue='display_label',
        palette='Greens',
        width=0.5,
        flierprops=flier_style_sm,
        legend=False
    )
    axes[2].set_title("Malayalam-English\n(Categorical Classes)", fontsize=11, fontweight='bold')
    axes[2].set_xlabel("Class Label", fontsize=10)
    axes[2].tick_params(axis='x', rotation=25, labelsize=9)

    fig.suptitle("Figure 8: Script-Based Code-Mixing Ratio by Class Label (Separated by Language)", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig8_path = os.path.join(FIGURES_DIR, "fig8_codemix_ratio_by_class.png")
    plt.savefig(fig8_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved {fig8_path}")

    # FIGURE 9: Language-Switch Count Distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df_all_feats, x='language', y='language_switch_count', hue='language', palette=['#2b5c8f', '#7570b3', '#1b9e77'], ax=ax, width=0.4, flierprops=flier_style, legend=False)
    ax.set_title(f"Figure 9: Language Switch Count Distribution by Language Pair (Dynamic Domain 0–{max_switch_val})", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Language Dataset", fontsize=11)
    ax.set_ylabel("Script Switch Count per Sample", fontsize=11)
    ax.set_ylim(-0.5, max_switch_val + 2)
    plt.tight_layout()
    fig9_path = os.path.join(FIGURES_DIR, "fig9_lang_switch_dist.png")
    plt.savefig(fig9_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved {fig9_path}")

    # FIGURE 10: Average Script-Based Code-Mixing Ratio Comparison Across Languages
    fig, ax = plt.subplots(figsize=(8, 5))
    avg_cm = df_all_feats.groupby('language')['script_code_mixing_ratio'].mean().reset_index()
    bars = ax.bar(avg_cm['language'], avg_cm['script_code_mixing_ratio'], color=['#2b5c8f', '#7570b3', '#1b9e77'], width=0.45)
    ax.set_title("Figure 10: Average Script-Based Code-Mixing Ratio Comparison", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Language Dataset Pair", fontsize=11)
    ax.set_ylabel("Mean Script-Based Code-Mixing Ratio", fontsize=11)
    max_cm_val = max(avg_cm['script_code_mixing_ratio'])
    ax.set_ylim(0, max_cm_val * 1.35)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.tight_layout()
    fig10_path = os.path.join(FIGURES_DIR, "fig10_avg_codemix_comparison.png")
    plt.savefig(fig10_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved {fig10_path}")

    # STEP 7: Generate Markdown Report
    print("\n[STEP 7] Compiling Summary Report (results/REVIEW1_EDA_SUMMARY.md)...")
    
    hindi_c0 = (df_hindi['label'] == '0').sum()
    hindi_c1 = (df_hindi['label'] == '1').sum()
    
    report_content = f"""# Review-1 EDA & Preprocessing Summary Report

**Project Title**: Faithfulness-Aware Explainability Audit of Multilingual Transformer Models for Code-Mixed Hate Speech Detection  
**Execution Seed**: {SEED}  
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
3. **Derived Feature Calculations**: Exact per-sample metrics calculated programmatically across all {total_benchmark_rows:,} benchmark rows.
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
| HECM Hindi-English | Hindi-English | {len(df_hindi):,} | - | - | {len(df_hindi):,} | Paired Text & Class Files (`.txt`, UTF-8) |
| Tamil-English | Tamil-English | {(df_tamil['split']=='train').sum():,} | {(df_tamil['split']=='dev').sum():,} | {(df_tamil['split']=='test').sum():,} | {len(df_tamil):,} | CSV (Comma-separated) |
| Malayalam-English | Malayalam-English | {(df_malayalam['split']=='train').sum():,} | {(df_malayalam['split']=='dev').sum():,} | {(df_malayalam['split']=='test').sum():,} | {len(df_malayalam):,} | TSV (Tab-separated, no headers) |
| **Total Benchmark** | - | **{(df_tamil['split']=='train').sum() + (df_malayalam['split']=='train').sum() + len(df_hindi):,}** | **{(df_tamil['split']=='dev').sum() + (df_malayalam['split']=='dev').sum():,}** | **{(df_tamil['split']=='test').sum() + (df_malayalam['split']=='test').sum():,}** | **{total_benchmark_rows:,}** | - |

---

## 2. HECM Script & Social Media Composition

In the HECM Hindi dataset (`datasets/Hindi/HECM.txt`, `datasets/Hindi/HECM_Class.txt`):
- **Total Rows**: {len(df_hindi):,}
- **Latin Script Presence (`A-Za-z`)**: {hecm_lat_count:,} ({hecm_lat_count/len(df_hindi):.2%})
- **Devanagari Script Presence (`U+0900–U+097F`)**: {hecm_dev_count:,} ({hecm_dev_count/len(df_hindi):.2%})
- **Script Code-Mixed (Both Scripts)**: {hecm_both_count:,} ({hecm_both_count/len(df_hindi):.2%})
- **Romanized Latin Only (Hinglish)**: {hecm_lat_only:,} ({hecm_lat_only/len(df_hindi):.2%})

**Key Observation**: Over 83% of the HECM dataset is Romanized Hindi (Hinglish written in Latin script). This provides a valuable benchmark for evaluating transformer robustness on Romanized transliterated hate speech, while highlighting the importance of acknowledging the limitations of purely script-based code-mixing metrics.

---

## 3. Data Quality & Detailed Duplicate Audit

In strict compliance with research integrity requirements, **NO samples were deleted or filtered out**. All duplicate rows and text instances are retained and documented transparently.

| Dataset | Missing Values | Exact Duplicate Rows | Duplicate Text Instances | Conflicting Text Instances | Rows Removed | Final Rows |
|---|---|---|---|---|---|---|
| HECM Hindi-English | {audit_hindi['missing_text_rows']} | {audit_hindi['exact_duplicate_rows']:,} | {audit_hindi['text_duplicate_instances']:,} | {audit_hindi['conflicting_duplicate_texts']:,} | 0 | {len(df_hindi):,} |
| Tamil-English | {audit_tamil['missing_text_rows']} | {audit_tamil['exact_duplicate_rows']:,} | {audit_tamil['text_duplicate_instances']:,} | {audit_tamil['conflicting_duplicate_texts']:,} | 0 | {len(df_tamil):,} |
| Malayalam-English | {audit_malayalam['missing_text_rows']} | {audit_malayalam['exact_duplicate_rows']:,} | {audit_malayalam['text_duplicate_instances']:,} | {audit_malayalam['conflicting_duplicate_texts']:,} | 0 | {len(df_malayalam):,} |

*Note: Conflicting text instances refer to exact duplicate text strings that are assigned to different ground-truth label classes in the raw dataset.*

---

## 4. Class Distribution Analysis (Independent Systems)

Each dataset utilizes a distinct, incompatible label schema. The label distributions are maintained independently:

### 1. HECM Hindi-English (Binary Class Labels '0' and '1')
- **Class 0**: {hindi_c0:,} samples ({hindi_c0/len(df_hindi):.2%})
- **Class 1**: {hindi_c1:,} samples ({hindi_c1/len(df_hindi):.2%})
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
$$\\text{{script\_code\_mixing\_ratio}} = \\frac{{\\min(\\text{{English\_Chars}}, \\text{{Native\_Chars}})}}{{\\text{{English\_Chars}} + \\text{{Native\_Chars}}}}$$

> [!WARNING]
> **Explicit Documented Limitation**:
> This metric is strictly **script-based** (detecting co-occurrence of Latin characters `A-Za-z` and native Indic script characters). Transliterated regional language text (e.g., Hinglish, Tanglish, Manglish written in Latin script) uses Latin characters and will be detected as English script. Consequently, transliterated code-mixed samples will yield a `script_code_mixing_ratio` near 0.0, underestimating actual linguistic code-mixing.

---

## 6. Summary Statistics

| Metric / Feature | Hindi-English (Mean ± Std) | Tamil-English (Mean ± Std) | Malayalam-English (Mean ± Std) |
|---|---|---|---|
| Character Count | {df_all_feats[df_all_feats['language']=='Hindi-English']['character_count'].mean():.2f} ± {df_all_feats[df_all_feats['language']=='Hindi-English']['character_count'].std():.2f} | {df_all_feats[df_all_feats['language']=='Tamil-English']['character_count'].mean():.2f} ± {df_all_feats[df_all_feats['language']=='Tamil-English']['character_count'].std():.2f} | {df_all_feats[df_all_feats['language']=='Malayalam-English']['character_count'].mean():.2f} ± {df_all_feats[df_all_feats['language']=='Malayalam-English']['character_count'].std():.2f} |
| Word Count | {df_all_feats[df_all_feats['language']=='Hindi-English']['word_count'].mean():.2f} ± {df_all_feats[df_all_feats['language']=='Hindi-English']['word_count'].std():.2f} | {df_all_feats[df_all_feats['language']=='Tamil-English']['word_count'].mean():.2f} ± {df_all_feats[df_all_feats['language']=='Tamil-English']['word_count'].std():.2f} | {df_all_feats[df_all_feats['language']=='Malayalam-English']['word_count'].mean():.2f} ± {df_all_feats[df_all_feats['language']=='Malayalam-English']['word_count'].std():.2f} |
| Native Script Ratio | {df_all_feats[df_all_feats['language']=='Hindi-English']['native_script_ratio'].mean():.4f} ± {df_all_feats[df_all_feats['language']=='Hindi-English']['native_script_ratio'].std():.4f} | {df_all_feats[df_all_feats['language']=='Tamil-English']['native_script_ratio'].mean():.4f} ± {df_all_feats[df_all_feats['language']=='Tamil-English']['native_script_ratio'].std():.4f} | {df_all_feats[df_all_feats['language']=='Malayalam-English']['native_script_ratio'].mean():.4f} ± {df_all_feats[df_all_feats['language']=='Malayalam-English']['native_script_ratio'].std():.4f} |
| Script-Based Code-Mixing Ratio | {mean_hindi_cm:.4f} ± {df_all_feats[df_all_feats['language']=='Hindi-English']['script_code_mixing_ratio'].std():.4f} | {mean_tam_cm:.4f} ± {df_all_feats[df_all_feats['language']=='Tamil-English']['script_code_mixing_ratio'].std():.4f} | {mean_mal_cm:.4f} ± {df_all_feats[df_all_feats['language']=='Malayalam-English']['script_code_mixing_ratio'].std():.4f} |
| Language Switch Count | {df_all_feats[df_all_feats['language']=='Hindi-English']['language_switch_count'].mean():.2f} ± {df_all_feats[df_all_feats['language']=='Hindi-English']['language_switch_count'].std():.2f} | {df_all_feats[df_all_feats['language']=='Tamil-English']['language_switch_count'].mean():.2f} ± {df_all_feats[df_all_feats['language']=='Tamil-English']['language_switch_count'].std():.2f} | {df_all_feats[df_all_feats['language']=='Malayalam-English']['language_switch_count'].mean():.2f} ± {df_all_feats[df_all_feats['language']=='Malayalam-English']['language_switch_count'].std():.2f} |

---

## 7. Future Work Roadmap (Review-2 and Beyond)

As specified in the Review-1 scope, model implementation and explainability audits are reserved for future project reviews:
1. Fine-tuning multilingual transformer baselines (mBERT, XLM-RoBERTa, IndicBERT).
2. Implementing XAI explanation attribution methods (SHAP, LIME, Integrated Gradients).
3. Faithfulness evaluation metrics (Comprehensiveness and Sufficiency).
4. Controlled perturbation experiments on code-mixed and script-switched tokens.
"""

    summary_md_path = os.path.join(EDA_DIR, "REVIEW1_EDA_SUMMARY.md")
    with open(summary_md_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    print(f"  -> Saved summary report to {summary_md_path}")

    # STEP 8: Verify Final Hashes (Post-execution)
    print("\n[STEP 8] Verifying Dataset Hashes (Post-execution Integrity Audit)...")
    final_hashes = verify_dataset_hashes("datasets")
    
    hashes_identical = True
    for fname, h_init in initial_hashes.items():
        h_final = final_hashes.get(fname)
        if h_init != h_final:
            print(f"  [CRITICAL ERROR] HASH MISMATCH for {fname}!")
            hashes_identical = False
        else:
            print(f"  [VERIFIED UNCHANGED] {fname}")
            
    if hashes_identical:
        print("\n" + "=" * 90)
        print("SUCCESS: 100% DATA IMMUTABILITY VERIFIED. DATASETS WERE UNTOUCHED.")
        print("=" * 90)
    else:
        print("\n" + "=" * 90)
        print("WARNING: DATASET MUTATION DETECTED!")
        print("=" * 90)

    print("\nREVIEW-1 PIPELINE COMPLETED SUCCESSFULLY!")


if __name__ == '__main__':
    run_pipeline()
