"""
src/preprocessing.py

Data loading, quality auditing, hashing verification, and transparent text preprocessing.

Strict Research Integrity Requirements:
- No raw dataset rows are deleted.
- Raw HECM Hindi text and class files are aligned 1-to-1 with raw string labels ('0', '1') preserved.
- Exact duplicate rows, duplicate text instances, and conflicting text instances are reported transparently.
- Malayalam is loaded with sep='\t', header=None, retaining column 0 as text and column 1 as label.
- Original files in datasets/ are NEVER modified.
"""

import os
import glob
import hashlib
import re
import unicodedata
import pandas as pd
import numpy as np


def compute_file_hash(filepath: str) -> str:
    """Computes SHA-256 hash of a file for integrity verification."""
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_dataset_hashes(base_dir: str = "datasets") -> dict:
    """
    Computes and returns SHA-256 hashes of all files in base_dir.
    Can be called before and after pipeline execution to ensure zero data mutation.
    """
    files = sorted(glob.glob(os.path.join(base_dir, "**", "*.*"), recursive=True))
    hash_dict = {}
    for filepath in files:
        if os.path.isfile(filepath):
            rel_path = os.path.relpath(filepath, base_dir)
            hash_dict[rel_path] = compute_file_hash(filepath)
    return hash_dict


def load_hindi(base_dir: str = "datasets/Hindi") -> pd.DataFrame:
    """
    Loads Hindi-English Code-Mixed (HECM) dataset from paired text and class files.
    Preserves exact 1-to-1 row alignment, raw labels ('0' and '1'), and sets language to 'Hindi-English'.
    """
    text_path = os.path.join(base_dir, "HECM.txt")
    if not os.path.isfile(text_path):
        text_path = os.path.join(base_dir, "HECM")

    class_path = os.path.join(base_dir, "HECM_Class.txt")
    if not os.path.isfile(class_path):
        class_path = os.path.join(base_dir, "HECM_Class")

    if not os.path.isfile(text_path):
        raise FileNotFoundError(f"HECM text file not found at: {text_path}")
    if not os.path.isfile(class_path):
        raise FileNotFoundError(f"HECM class file not found at: {class_path}")

    with open(text_path, 'r', encoding='utf-8') as f:
        texts = [line.rstrip('\r\n') for line in f]

    with open(class_path, 'r', encoding='utf-8') as f:
        classes = [line.rstrip('\r\n') for line in f]

    if len(texts) != len(classes):
        raise ValueError(
            f"Row count mismatch in HECM dataset: {len(texts)} text samples vs {len(classes)} class labels."
        )

    # Validate that all labels are string '0' or '1'
    invalid_labels = set(classes) - {'0', '1'}
    if invalid_labels:
        raise ValueError(f"Unexpected class label(s) found in HECM_Class: {invalid_labels}")

    df = pd.DataFrame({
        'text': texts,
        'label': classes,
        'language': 'Hindi-English',
        'split': 'full'
    })
    return df


def load_tamil(base_dir: str = "datasets/TAMIL") -> pd.DataFrame:
    """
    Loads Tamil-English dataset (train, dev, test).
    Columns: 'text', 'label'.
    """
    splits = []
    for split_name in ['train', 'dev', 'test']:
        filepath = os.path.join(base_dir, f"{split_name}.csv")
        df_split = pd.read_csv(filepath)
        df_split['language'] = 'Tamil-English'
        df_split['split'] = split_name
        splits.append(df_split)
    
    df_tamil = pd.concat(splits, ignore_index=True)
    return df_tamil


def load_malayalam(base_dir: str = "datasets/MALAYALAM") -> pd.DataFrame:
    """
    Loads Malayalam-English dataset (train, dev, test).
    Files are tab-separated with no headers.
    Column 0 = text, Column 1 = label.
    Retains column 0 as text and column 1 as label.
    """
    splits = []
    for split_name in ['train', 'dev', 'test']:
        filepath = os.path.join(base_dir, f"{split_name}.csv")
        df_raw = pd.read_csv(filepath, sep='\t', header=None)
        # Select first two columns explicitly to handle trailing tabs
        df_split = df_raw.iloc[:, :2].copy()
        df_split.columns = ['text', 'label']
        df_split['language'] = 'Malayalam-English'
        df_split['split'] = split_name
        splits.append(df_split)
    
    df_malayalam = pd.concat(splits, ignore_index=True)
    return df_malayalam


def load_all_datasets(datasets_dir: str = "datasets") -> pd.DataFrame:
    """Loads all three language datasets and returns a combined DataFrame."""
    df_hindi = load_hindi(os.path.join(datasets_dir, "Hindi"))
    df_tamil = load_tamil(os.path.join(datasets_dir, "TAMIL"))
    df_malayalam = load_malayalam(os.path.join(datasets_dir, "MALAYALAM"))
    
    # Common columns for uniform structure
    cols = ['text', 'label', 'language', 'split']
    df_combined = pd.concat([
        df_hindi[cols],
        df_tamil[cols],
        df_malayalam[cols]
    ], ignore_index=True)
    
    return df_combined


def compute_duplicate_audit(df: pd.DataFrame) -> dict:
    """
    Audits exact duplicate rows, duplicate text instances, and conflicting text instances.
    Conflicting duplicates = same text assigned to different label classes.
    """
    total_rows = len(df)
    missing_text = df['text'].isna().sum()
    
    # Fill NA temporarily for grouping if any
    df_clean = df.copy()
    df_clean['text'] = df_clean['text'].fillna('')
    
    exact_duplicates = df_clean.duplicated().sum()
    text_duplicates = df_clean.duplicated(subset=['text']).sum()
    
    # Conflicting duplicate texts
    text_label_counts = df_clean.groupby('text')['label'].nunique()
    conflicting_texts = (text_label_counts > 1).sum()
    
    return {
        "total_rows": total_rows,
        "missing_text_rows": int(missing_text),
        "exact_duplicate_rows": int(exact_duplicates),
        "text_duplicate_instances": int(text_duplicates),
        "conflicting_duplicate_texts": int(conflicting_texts),
        "rows_removed": 0,
        "final_rows": total_rows
    }


def preprocess_text(text: str) -> str:
    """
    Transparent, non-destructive text preprocessing function.
    
    Operations performed:
    1. Unicode NFC Normalization: Standardizes Unicode composite characters.
    2. Whitespace Normalization: Collapses multiple spaces, tabs, line breaks into single space.
    3. Consistent URL handling: Standardizes URLs to '[URL]'.
    4. Consistent User Mention handling: Standardizes '@user' handles to '[USER]'.
    5. Removal of invisible formatting noise: Strips zero-width spaces (\\u200c, \\u200d).
    
    Preserved intact:
    - English words, native scripts (Devanagari, Tamil, Malayalam), code-mixed tokens.
    - Numbers, hashtags, and all meaningful punctuation.
    - NO translation, NO transliteration, NO stop-word removal, NO stemming/lemmatization.
    - NO rows deleted.
    """
    if not isinstance(text, str) or pd.isna(text):
        return ""
    
    # 1. Unicode NFC Normalization
    text = unicodedata.normalize('NFC', text)
    
    # 2. Strip invisible zero-width spaces
    text = text.replace('\u200c', '').replace('\u200d', '')
    
    # 3. Standardize URLs
    text = re.sub(r'http[s]?://\S+|www\.\S+', '[URL]', text)
    
    # 4. Standardize User Mentions
    text = re.sub(r'@\w+', '[USER]', text)
    
    # 5. Whitespace Normalization
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text
