"""
src/robustness/script_composition.py

Analysis module for Unicode script composition and code-mixing robustness.

Linguistic & Methodological Caveat:
- Script composition is determined by Unicode character code points (Latin vs Native Indic scripts).
- This is a surface script-level metric and is NOT equivalent to true linguistic code-mixing.
- Transliterated / Romanized code-mixed speech (e.g., Hinglish or Tanglish written purely in
  the Latin alphabet) contains only Latin characters and will be categorized as Single-script (Latin).
- This limitation must be maintained in all reporting and publication documentation.
"""

import os
from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
DEFAULT_METRICS_CSV = os.path.join(PROJECT_ROOT, "results", "robustness", "script_composition_per_class_metrics.csv")


def classify_character_script(char: str, language_pair: str) -> str:
    """
    Classifies a character into:
    - 'Latin': Latin characters (A-Z, a-z)
    - 'Native': Indic script matching language_pair (Devanagari, Tamil, Malayalam)
    - 'Other': Whitespace, punctuation, numbers, symbols
    """
    code_point = ord(char)
    
    # Latin alphabet
    if (65 <= code_point <= 90) or (97 <= code_point <= 122):
        return "Latin"
        
    lang = language_pair.lower()
    if "hindi" in lang or "hecm" in lang:
        # Devanagari Unicode Block: U+0900 to U+097F
        if 0x0900 <= code_point <= 0x097F:
            return "Native"
    elif "tamil" in lang:
        # Tamil Unicode Block: U+0B80 to U+0BFF
        if 0x0B80 <= code_point <= 0x0BFF:
            return "Native"
    elif "malayalam" in lang:
        # Malayalam Unicode Block: U+0D00 to U+0DFF
        if 0x0D00 <= code_point <= 0x0DFF:
            return "Native"
            
    return "Other"


def classify_text_script_composition(text: str, language_pair: str) -> str:
    """
    Categorizes text as:
    - 'Mixed-script': contains BOTH Latin and Native Indic script characters
    - 'Single-script': contains only Latin or only Native script (or neither)
    """
    if not isinstance(text, str) or not text.strip():
        return "Single-script"
        
    has_latin = False
    has_native = False
    
    for c in text:
        script = classify_character_script(c, language_pair)
        if script == "Latin":
            has_latin = True
        elif script == "Native":
            has_native = True
            
        if has_latin and has_native:
            return "Mixed-script"
            
    return "Single-script"


def load_authoritative_per_class_metrics(csv_path: Optional[str] = None) -> pd.DataFrame:
    """Loads authoritative script_composition_per_class_metrics.csv."""
    path = csv_path or DEFAULT_METRICS_CSV
    if not os.path.exists(path):
        raise FileNotFoundError(f"Authoritative per-class metrics CSV not found at: {path}")
    return pd.read_csv(path)


def compute_composition_macro_f1_gap(csv_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Calculates Macro-F1 performance degradation from Single-script to Mixed-script
    for each language from authoritative per-class metrics.
    """
    df = load_authoritative_per_class_metrics(csv_path)
    
    results = {}
    for lang, lang_df in df.groupby("language"):
        single_df = lang_df[lang_df["composition"] == "Single-script"]
        mixed_df = lang_df[lang_df["composition"] == "Mixed-script"]
        
        single_macro_f1 = float(single_df["f1"].mean())
        mixed_macro_f1 = float(mixed_df["f1"].mean())
        gap = single_macro_f1 - mixed_macro_f1
        
        single_support = int(single_df["support"].sum())
        mixed_support = int(mixed_df["support"].sum())
        
        results[lang] = {
            "single_script_macro_f1": single_macro_f1,
            "single_script_support": single_support,
            "mixed_script_macro_f1": mixed_macro_f1,
            "mixed_script_support": mixed_support,
            "f1_performance_drop": gap,
            "relative_drop_pct": (gap / single_macro_f1 * 100) if single_macro_f1 > 0 else 0.0
        }
        
    return results
