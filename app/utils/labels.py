"""
app/utils/labels.py

Authoritative label mappings, descriptions, and representative sample inputs
for the Team 6 Streamlit demonstration interface.

Research Constraints:
- HECM: 0 = Hate, 1 = Non-Hate
- Tamil: 0 to 5 preserving authoritative project definitions
- Malayalam: 5 categorical string labels preserving exact dataset terms
- Representative samples for each language pair for zero-barrier testing
"""

from typing import Dict, List, Any

# 1. Label Mappings
LABEL_MAPPINGS: Dict[str, Dict[int, str]] = {
    "hecm": {
        0: "Hate",
        1: "Non-Hate"
    },
    "tamil": {
        0: "Not-offensive (label 0)",
        1: "Offensive_Category_1 (label 1)",
        2: "Offensive_Category_2 (label 2)",
        3: "Offensive_Category_3 (label 3)",
        4: "Offensive_Category_4 (label 4)",
        5: "Offensive_Category_5 (label 5)"
    },
    "malayalam": {
        0: "Not_offensive",
        1: "not-malayalam",
        2: "Offensive_Targeted_Insult_Individual",
        3: "Offensive_Untargetede",  # Preserved dataset spelling
        4: "Offensive_Targeted_Insult_Group"
    }
}

# Compact label tags for charts and badges
LABEL_SHORT_NAMES: Dict[str, Dict[int, str]] = {
    "hecm": {
        0: "Hate",
        1: "Non-Hate"
    },
    "tamil": {
        0: "Label 0 (Not-offensive)",
        1: "Label 1",
        2: "Label 2 (Counter-speech)",
        3: "Label 3",
        4: "Label 4",
        5: "Label 5"
    },
    "malayalam": {
        0: "Not_offensive",
        1: "not-malayalam",
        2: "Insult_Individual",
        3: "Offensive_Untargeted",
        4: "Insult_Group"
    }
}

# 2. Representative sample inputs from validated benchmark sets
SAMPLE_TEXTS: Dict[str, List[Dict[str, str]]] = {
    "hecm": [
        {
            "title": "Hindi-English Sample 1 (Non-Hate / Political debate)",
            "text": "Modi ji ne farmers ke liye kafi policy banayi hain par implementation thoda slow hai."
        },
        {
            "title": "Hindi-English Sample 2 (Hate speech / Insult)",
            "text": "Abe gadhe chup baith, tum sab desh drohi log ho or jhooth bol rahe ho."
        },
        {
            "title": "Hindi-English Sample 3 (Mixed-Script Devanagari + Latin)",
            "text": "Yeh bilkul galat baat hai भाई, aisa mat karo please don't spread hate."
        }
    ],
    "tamil": [
        {
            "title": "Tamil Sample 1 (Romanized / Positive trailer reaction)",
            "text": "movie vara level la Erika poguthu sema trailer thalaiva"
        },
        {
            "title": "Tamil Sample 2 (Native Script / Wishes)",
            "text": "தென்காசி மாவட்டம் நாடார் சமுதாயம் சார்பாக வாழ்த்துக்கள்"
        },
        {
            "title": "Tamil Sample 3 (Code-Mixed YouTube Comment)",
            "text": "Thala mass scene pakka theatre la fire vidum super video"
        }
    ],
    "malayalam": [
        {
            "title": "Malayalam Sample 1 (Native Script / Film discussion)",
            "text": "പലദേശം പല ഭാഷ ഒരേ ഒരു രാജാവ് സ്വന്തം രാജ്യം അല്ല"
        },
        {
            "title": "Malayalam Sample 2 (Mixed Script / Music reaction)",
            "text": "Gopichettante BGM um mammookayum ishtapedunnavar like adikku"
        },
        {
            "title": "Malayalam Sample 3 (Aggressive comment)",
            "text": "enth oola editing aado ith oru madhiri verupikkal video"
        }
    ]
}


def get_labels_for_dataset(dataset_key: str) -> Dict[int, str]:
    """Returns id-to-label mapping for the specified dataset key."""
    key = dataset_key.lower()
    return LABEL_MAPPINGS.get(key, {})


def get_label_name(dataset_key: str, class_idx: int) -> str:
    """Returns the descriptive label for a specific class index."""
    key = dataset_key.lower()
    mapping = LABEL_MAPPINGS.get(key, {})
    return mapping.get(class_idx, f"Class {class_idx}")


def get_label_short_name(dataset_key: str, class_idx: int) -> str:
    """Returns compact display label for a specific class index."""
    key = dataset_key.lower()
    mapping = LABEL_SHORT_NAMES.get(key, {})
    return mapping.get(class_idx, f"C{class_idx}")


def get_samples_for_dataset(dataset_key: str) -> List[Dict[str, str]]:
    """Returns sample test inputs for the specified dataset key."""
    key = dataset_key.lower()
    return SAMPLE_TEXTS.get(key, [])
