"""
src/models/model_registry.py

Authoritative model registry, checkpoint resolution, label mappings,
and inference loader utilities for the Team6_XAI project.

Core Research Principles:
- Models are bert-base-multilingual-cased fine-tuned in Google Colab.
- Authoritative inference checkpoints are stored under results/modeling/.
- Training-resume state was excluded; checkpoints are for inference and audit only.
- NO model training functionality is provided in this registry.
"""

import os
from typing import Dict, Any, Optional, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Base directories
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))

# Selected foundation model
BASE_MODEL_NAME = "bert-base-multilingual-cased"
FINAL_MODEL_ARCH = "mBERT"

# Comparative models evaluated in initial validation (historical record)
COMPARATIVE_MODELS = {
    "xlm-r": "xlm-roberta-base",
    "indic-bert": "ai4bharat/indic-bert"
}

# Checkpoint paths relative to PROJECT_ROOT
CHECKPOINT_PATHS = {
    "hecm": os.path.join("results", "modeling", "mbert_hecm", "checkpoint-824"),
    "tamil": os.path.join("results", "modeling", "mbert_tamil", "checkpoint-3844"),
    "malayalam": os.path.join("results", "modeling", "mbert_malayalam", "checkpoint-1756")
}

# Number of classification classes per task
NUM_LABELS = {
    "hecm": 2,
    "tamil": 6,
    "malayalam": 5
}

# Authoritative label mappings
# 1. HECM: binary classification ('0' -> Hateful, '1' -> Non-Hate)
HECM_LABEL2ID = {"0": 0, "1": 1}
HECM_ID2LABEL = {0: "0", 1: "1"}
HECM_LABEL_DESCRIPTIONS = {0: "Hate", 1: "Non-Hate"}

# 2. Tamil: 6-class classification (0 to 5)
TAMIL_LABEL2ID = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
TAMIL_ID2LABEL = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
TAMIL_LABEL_DESCRIPTIONS = {
    0: "Not-offensive",
    1: "Offensive_Category_1",
    2: "Offensive_Category_2",
    3: "Offensive_Category_3",
    4: "Offensive_Category_4",
    5: "Offensive_Category_5"
}

# 3. Malayalam: 5 categorical string labels
MALAYALAM_LABEL2ID = {
    "Not_offensive": 0,
    "not-malayalam": 1,
    "Offensive_Targeted_Insult_Individual": 2,
    "Offensive_Untargetede": 3,  # Original dataset typo preserved
    "Offensive_Targeted_Insult_Group": 4
}
MALAYALAM_ID2LABEL = {v: k for k, v in MALAYALAM_LABEL2ID.items()}
MALAYALAM_LABEL_DESCRIPTIONS = dict(MALAYALAM_ID2LABEL)

# Sequence length & audit configurations
MAX_SEQ_LENGTH = 128
EVAL_BATCH_SIZE = 32
XAI_AUDIT_SAMPLE_SIZE = 100
RANDOM_SEED = 42

# Expected test set sizes (strict verification targets)
EXPECTED_TEST_SIZES = {
    "hecm": 1398,
    "tamil": 6584,
    "malayalam": 2997
}


def get_checkpoint_dir(dataset_key: str) -> str:
    """Returns absolute path to the authoritative checkpoint directory."""
    key = dataset_key.lower()
    if key not in CHECKPOINT_PATHS:
        raise KeyError(f"Unknown dataset key '{dataset_key}'. Must be one of {list(CHECKPOINT_PATHS.keys())}")
    rel_path = CHECKPOINT_PATHS[key]
    abs_path = os.path.join(PROJECT_ROOT, rel_path)
    if not os.path.isdir(abs_path):
        raise FileNotFoundError(f"Checkpoint directory for '{dataset_key}' not found at: {abs_path}")
    return abs_path


def get_model_spec(dataset_key: str) -> Dict[str, Any]:
    """Returns full specification dictionary for a registered model."""
    key = dataset_key.lower()
    if key not in CHECKPOINT_PATHS:
        raise KeyError(f"Unknown dataset key '{dataset_key}'. Must be one of {list(CHECKPOINT_PATHS.keys())}")
    
    ckpt_dir = get_checkpoint_dir(key)
    
    if key == "hecm":
        l2id, id2l = HECM_LABEL2ID, HECM_ID2LABEL
        desc = HECM_LABEL_DESCRIPTIONS
    elif key == "tamil":
        l2id, id2l = TAMIL_LABEL2ID, TAMIL_ID2LABEL
        desc = TAMIL_LABEL_DESCRIPTIONS
    else:
        l2id, id2l = MALAYALAM_LABEL2ID, MALAYALAM_ID2LABEL
        desc = MALAYALAM_LABEL_DESCRIPTIONS
        
    return {
        "dataset_key": key,
        "base_model": BASE_MODEL_NAME,
        "model_architecture": FINAL_MODEL_ARCH,
        "checkpoint_dir": ckpt_dir,
        "num_labels": NUM_LABELS[key],
        "label2id": l2id,
        "id2label": id2l,
        "label_descriptions": desc,
        "max_seq_length": MAX_SEQ_LENGTH,
        "eval_batch_size": EVAL_BATCH_SIZE,
        "expected_test_size": EXPECTED_TEST_SIZES[key]
    }


def load_tokenizer(dataset_key: str) -> AutoTokenizer:
    """Loads tokenizer from the authoritative checkpoint directory."""
    ckpt_dir = get_checkpoint_dir(dataset_key)
    return AutoTokenizer.from_pretrained(ckpt_dir)


def load_model(dataset_key: str, device: Optional[str] = None) -> AutoModelForSequenceClassification:
    """
    Loads fine-tuned mBERT classification model from the authoritative checkpoint.
    Sets model to eval() mode.
    """
    ckpt_dir = get_checkpoint_dir(dataset_key)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir)
    model.eval()
    if device is not None:
        model.to(torch.device(device))
    return model


def load_model_and_tokenizer(
    dataset_key: str, device: Optional[str] = None
) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """Convenience helper to load both model and tokenizer."""
    tok = load_tokenizer(dataset_key)
    mod = load_model(dataset_key, device=device)
    return mod, tok
