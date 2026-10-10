"""
src/data/loaders.py

Reusable utilities for loading datasets and applying leakage-safe,
stratified train/validation/test splits for HECM, Tamil, and Malayalam.

Strict Research Integrity Requirements:
- Uses existing src.data.preprocessing functions without data modification.
- Never generates new random splits; reads only authoritative split JSONs.
- Verifies zero cross-split index and group leakage.
"""

import os
import json
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd

from src.data.preprocessing import (
    load_hindi,
    load_tamil,
    load_malayalam,
    preprocess_text
)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))

SPLIT_PATHS = {
    "hecm": os.path.join(PROJECT_ROOT, "results", "modeling", "splits", "hecm_split_seed42.json"),
    "tamil": os.path.join(PROJECT_ROOT, "results", "modeling", "splits", "tamil_split_seed42.json"),
    "malayalam": os.path.join(PROJECT_ROOT, "results", "modeling", "splits", "malayalam_split_seed42.json")
}

EXPECTED_COUNTS = {
    "hecm": {
        "total": 9357,
        "train": 6561,
        "validation": 1398,
        "test": 1398
    },
    "tamil": {
        "total": 43919,
        "train": 30748,
        "validation": 6587,
        "test": 6584
    },
    "malayalam": {
        "total": 20010,
        "train": 14033,
        "validation": 2980,
        "test": 2997
    }
}


def load_raw_dataset(dataset_key: str) -> pd.DataFrame:
    """Loads the full raw dataset before split partitioning."""
    key = dataset_key.lower()
    if key == "hecm":
        return load_hindi(os.path.join(PROJECT_ROOT, "datasets", "Hindi"))
    elif key == "tamil":
        return load_tamil(os.path.join(PROJECT_ROOT, "datasets", "TAMIL"))
    elif key == "malayalam":
        return load_malayalam(os.path.join(PROJECT_ROOT, "datasets", "MALAYALAM"))
    else:
        raise ValueError(f"Unknown dataset_key '{dataset_key}'. Must be 'hecm', 'tamil', or 'malayalam'.")


def load_split_metadata(dataset_key: str) -> Dict[str, Any]:
    """Loads the raw metadata stored in the authoritative split JSON."""
    key = dataset_key.lower()
    if key not in SPLIT_PATHS:
        raise KeyError(f"Unknown dataset_key '{dataset_key}'. Available: {list(SPLIT_PATHS.keys())}")
    path = SPLIT_PATHS[key]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Split JSON not found at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_split_indices(dataset_key: str) -> Dict[str, List[int]]:
    """
    Extracts standardized {'train': [...], 'validation': [...], 'test': [...]}
    indices from the authoritative split JSON.
    """
    meta = load_split_metadata(dataset_key)
    if "indices" in meta:
        raw_indices = meta["indices"]
        return {
            "train": list(raw_indices["train"]),
            "validation": list(raw_indices["validation"]),
            "test": list(raw_indices["test"])
        }
    elif "train_indices" in meta:
        return {
            "train": list(meta["train_indices"]),
            "validation": list(meta["validation_indices"]),
            "test": list(meta["test_indices"])
        }
    else:
        raise KeyError(f"Could not identify index keys in split JSON for '{dataset_key}'.")


def validate_split_integrity(dataset_key: str) -> Dict[str, Any]:
    """
    Performs comprehensive verification of the split:
    - Index non-overlap (zero index leakage across train, val, test)
    - Sample count matches expected authoritative counts
    - Text group non-overlap (zero group leakage into test)
    """
    key = dataset_key.lower()
    indices = load_split_indices(key)
    df = load_raw_dataset(key)
    
    train_idx = set(indices["train"])
    val_idx = set(indices["validation"])
    test_idx = set(indices["test"])
    
    # Check 1: Zero index overlap
    train_val_overlap = len(train_idx & val_idx)
    train_test_overlap = len(train_idx & test_idx)
    val_test_overlap = len(val_idx & test_idx)
    zero_index_leakage = (train_val_overlap == 0 and train_test_overlap == 0 and val_test_overlap == 0)
    
    # Check 2: Sample count parity
    expected = EXPECTED_COUNTS[key]
    counts_match = (
        len(df) == expected["total"] and
        len(indices["train"]) == expected["train"] and
        len(indices["validation"]) == expected["validation"] and
        len(indices["test"]) == expected["test"]
    )
    
    # Check 3: Text overlap between train and test
    train_texts = set(df.iloc[indices["train"]]["text"].dropna())
    test_texts = set(df.iloc[indices["test"]]["text"].dropna())
    text_overlap_count = len(train_texts & test_texts)
    
    return {
        "dataset_key": key,
        "total_rows": len(df),
        "train_rows": len(indices["train"]),
        "validation_rows": len(indices["validation"]),
        "test_rows": len(indices["test"]),
        "train_val_overlap": train_val_overlap,
        "train_test_overlap": train_test_overlap,
        "val_test_overlap": val_test_overlap,
        "zero_index_leakage": zero_index_leakage,
        "counts_match": counts_match,
        "text_train_test_overlap": text_overlap_count,
        "integrity_passed": (zero_index_leakage and counts_match and text_overlap_count == 0)
    }


def load_split_data(
    dataset_key: str,
    split_name: str = "test",
    preprocess: bool = False
) -> pd.DataFrame:
    """
    Loads a specific split ('train', 'validation', or 'test') as a DataFrame.
    Optionally applies standard non-destructive preprocessing.
    """
    name = "validation" if split_name in ("val", "dev") else split_name
    indices = load_split_indices(dataset_key)
    if name not in indices:
        raise ValueError(f"Invalid split_name '{split_name}'. Must be 'train', 'validation', or 'test'.")
    
    df = load_raw_dataset(dataset_key)
    split_df = df.iloc[indices[name]].copy().reset_index(drop=True)
    split_df["split"] = name
    
    if preprocess:
        split_df["text"] = split_df["text"].apply(preprocess_text)
        
    return split_df


def load_dataset_with_splits(
    dataset_key: str,
    preprocess: bool = False
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Loads full dataset and a dictionary of partitioned DataFrames:
    {'train': df_train, 'validation': df_val, 'test': df_test}.
    """
    key = dataset_key.lower()
    full_df = load_raw_dataset(key)
    splits = {
        "train": load_split_data(key, "train", preprocess=preprocess),
        "validation": load_split_data(key, "validation", preprocess=preprocess),
        "test": load_split_data(key, "test", preprocess=preprocess)
    }
    return full_df, splits
