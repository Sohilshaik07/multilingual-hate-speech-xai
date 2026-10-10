"""
src/feature_engineering.py

Extraction of 13 statistical, script-level, and code-mixing features.

IMPORTANT DOCUMENTATION ON METRICS & LIMITATIONS:
1. Metric Name: 'script_code_mixing_ratio'
   Definition: min(english_character_count, native_script_character_count) / 
               (english_character_count + native_script_character_count)

2. Explicit Documented Limitation:
   This metric is purely script-based. It detects code-mixing via co-occurrence of 
   Latin (English) script and native Indic scripts (Devanagari, Tamil, Malayalam).
   Transliterated regional language text (e.g., Hinglish, Tanglish, Manglish written in Latin script)
   uses Latin characters and will be categorized as English-script characters, resulting in an 
   underestimation of code-mixing by this script-based metric.
"""

import re
import string
import unicodedata
import pandas as pd
import numpy as np


def get_character_script(char: str, language_pair: str) -> str:
    """
    Returns the script classification for a single character:
    'E': English (Latin A-Z, a-z)
    'N': Native script (Devanagari for Hindi, Tamil for Tamil, Malayalam for Malayalam)
    'O': Other (digits, spaces, punctuation, symbols)
    """
    code_point = ord(char)
    
    # English (Latin A-Z, a-z)
    if (65 <= code_point <= 90) or (97 <= code_point <= 122):
        return 'E'
    
    # Language-specific Native Script ranges
    if language_pair == 'Hindi-English':
        # Devanagari range: U+0900 to U+097F
        if 0x0900 <= code_point <= 0x097F:
            return 'N'
    elif language_pair == 'Tamil-English':
        # Tamil range: U+0B80 to U+0BFF
        if 0x0B80 <= code_point <= 0x0BFF:
            return 'N'
    elif language_pair == 'Malayalam-English':
        # Malayalam range: U+0D00 to U+0DFF
        if 0x0D00 <= code_point <= 0x0DFF:
            return 'N'
    
    return 'O'


def count_language_switches(text: str, language_pair: str) -> int:
    """
    Calculates transitions between English-script characters and Native-script characters,
    ignoring spaces, digits, and punctuation.
    """
    if not isinstance(text, str) or not text:
        return 0
    
    # Filter text to sequence of script tokens ('E' or 'N')
    script_seq = []
    for c in text:
        s = get_character_script(c, language_pair)
        if s in ('E', 'N'):
            script_seq.append(s)
    
    if len(script_seq) <= 1:
        return 0
    
    switches = 0
    for i in range(1, len(script_seq)):
        if script_seq[i] != script_seq[i - 1]:
            switches += 1
            
    return switches


def extract_sample_features(text: str, language_pair: str) -> dict:
    """
    Extracts all 13 interpretable linguistic and script-based code-mixing features 
    for a single text sample.
    """
    if not isinstance(text, str) or pd.isna(text):
        text = ""
    
    # 1. character_count
    char_count = len(text)
    
    # 2. word_count
    words = text.split()
    word_count = len(words)
    
    # 3. sentence_count (delimited by . ! ? | ।)
    sentences = [s for s in re.split(r'[.!?|\u0964]+', text) if s.strip()]
    sentence_count = max(1, len(sentences)) if text.strip() else 0
    
    # 4. average_word_length
    avg_word_len = char_count / word_count if word_count > 0 else 0.0
    
    # 5. digit_count
    digit_count = sum(1 for c in text if c.isdigit())
    
    # 6. punctuation_count (ASCII + native punctuation like danda U+0964)
    punct_count = sum(1 for c in text if c in string.punctuation or c == '\u0964')
    
    # 7. uppercase_ratio
    letters = [c for c in text if c.isalpha()]
    total_letters = len(letters)
    uppercase_count = sum(1 for c in text if c.isupper())
    uppercase_ratio = uppercase_count / total_letters if total_letters > 0 else 0.0
    
    # 8. english_character_count & 9. native_script_character_count
    english_char_count = 0
    native_char_count = 0
    for c in text:
        s = get_character_script(c, language_pair)
        if s == 'E':
            english_char_count += 1
        elif s == 'N':
            native_char_count += 1
            
    total_script_chars = english_char_count + native_char_count
    
    # 10. english_ratio
    english_ratio = english_char_count / total_script_chars if total_script_chars > 0 else 0.0
    
    # 11. native_script_ratio
    native_script_ratio = native_char_count / total_script_chars if total_script_chars > 0 else 0.0
    
    # 12. language_switch_count
    switch_count = count_language_switches(text, language_pair)
    
    # 13. script_code_mixing_ratio
    script_cm_ratio = min(english_char_count, native_char_count) / total_script_chars if total_script_chars > 0 else 0.0
    
    return {
        "character_count": char_count,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "average_word_length": round(avg_word_len, 4),
        "digit_count": digit_count,
        "punctuation_count": punct_count,
        "uppercase_ratio": round(uppercase_ratio, 4),
        "english_character_count": english_char_count,
        "native_script_character_count": native_char_count,
        "english_ratio": round(english_ratio, 4),
        "native_script_ratio": round(native_script_ratio, 4),
        "language_switch_count": switch_count,
        "script_code_mixing_ratio": round(script_cm_ratio, 4)
    }


def extract_features(df: pd.DataFrame, text_col: str = 'text', lang_col: str = 'language') -> pd.DataFrame:
    """
    Computes and appends all 13 features to the DataFrame.
    """
    features_list = []
    for idx, row in df.iterrows():
        text = row[text_col]
        lang = row[lang_col]
        feats = extract_sample_features(text, lang)
        features_list.append(feats)
        
    df_feats = pd.DataFrame(features_list)
    df_result = pd.concat([df.reset_index(drop=True), df_feats], axis=1)
    return df_result


def get_feature_definitions() -> pd.DataFrame:
    """Returns a DataFrame summarizing feature definitions and analysis purposes."""
    features_info = [
        ("character_count", "Total number of characters in raw/preprocessed text", "Text length analysis"),
        ("word_count", "Total space-separated words/tokens in text", "Token density and document length"),
        ("sentence_count", "Estimated sentence count based on delimiters (.!?|।)", "Structural complexity"),
        ("average_word_length", "Ratio of character count to word count", "Vocabulary complexity"),
        ("digit_count", "Count of numeric characters", "Presence of numbers/years/counts"),
        ("punctuation_count", "Count of ASCII and native punctuation marks", "Syntactic noise and style"),
        ("uppercase_ratio", "Ratio of uppercase Latin letters to total letters", "Emphasis/shouting detection"),
        ("english_character_count", "Count of Latin script alphabetic characters (A-Z, a-z)", "English/Roman script density"),
        ("native_script_character_count", "Count of native Indic script characters (Devanagari/Tamil/Malayalam)", "Native script density"),
        ("english_ratio", "Ratio of English characters to total script characters", "Proportion of Latin script"),
        ("native_script_ratio", "Ratio of native script characters to total script characters", "Proportion of native script"),
        ("language_switch_count", "Number of transitions between English script and native script", "Script-level switching rate"),
        ("script_code_mixing_ratio", "min(English, Native) / (English + Native)", "Script-level co-occurrence of code-mixing (Note: Underestimates Latin-transliterated regional text)")
    ]
    return pd.DataFrame(features_info, columns=["Feature", "Definition", "Purpose"])
