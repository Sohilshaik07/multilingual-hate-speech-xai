"""
app/utils/model_loader.py

Cached model and tokenizer loaders and inference utilities for the Streamlit demo.
Integrates directly with src/models/model_registry.py.

Constraints:
- Uses @st.cache_resource to prevent redundant checkpoint reloading.
- Loads mBERT checkpoints with eager attention implementation for attention extraction.
- Defaults to CPU execution for stable deployment.
- Truncation-safe: enforces 126 content tokens, 128 max sequence length.

Model Source Selection:
- Set environment variable MODEL_SOURCE=hf   → load from HuggingFace Hub (default for cloud)
- Set environment variable MODEL_SOURCE=local → load from local checkpoint directories

HuggingFace Repository Map:
  hecm:      sohilshaik07/team6-mbert-hecm
  tamil:     sohilshaik07/team6-mbert-tamil
  malayalam: sohilshaik07/team6-mbert-malayalam
"""

import os
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.models.model_registry import (
    get_checkpoint_dir,
    get_model_spec,
    MAX_SEQ_LENGTH,
    CHECKPOINT_PATHS,
    PROJECT_ROOT,
)
from src.xai.shap_audit import get_model_visible_text
try:
    from app.utils.labels import get_labels_for_dataset
except (ModuleNotFoundError, ImportError):
    from utils.labels import get_labels_for_dataset


# ---------------------------------------------------------------------------
# HuggingFace Hub repository IDs for the three validated checkpoints
# ---------------------------------------------------------------------------
HF_REPO_IDS: Dict[str, str] = {
    "hecm":      "sohilshaik07/team6-mbert-hecm",
    "tamil":     "sohilshaik07/team6-mbert-tamil",
    "malayalam": "sohilshaik07/team6-mbert-malayalam",
}


def get_model_source() -> str:
    """
    Determines the model source from the MODEL_SOURCE environment variable.

    Returns:
        'hf'    — load from HuggingFace Hub (default when env var is absent or empty)
        'local' — load from local checkpoint directories

    Raises:
        ValueError: If MODEL_SOURCE is set to an unsupported value.
    """
    raw = os.environ.get("MODEL_SOURCE", "hf")
    source = raw.strip().lower() if raw and raw.strip() else "hf"
    if source not in ("hf", "local"):
        raise ValueError(
            f"Invalid MODEL_SOURCE '{raw}'. Supported options are 'hf' or 'local'."
        )
    return source


def get_model_identifier(dataset_key: str) -> str:
    """
    Returns the model identifier string (HF repo ID or local path) for the
    given dataset key, based on the active MODEL_SOURCE.
    """
    key = dataset_key.lower()
    source = get_model_source()
    if source == "local":
        return str(get_checkpoint_dir(key))
    else:
        repo_id = HF_REPO_IDS.get(key)
        if repo_id is None:
            raise ValueError(
                f"No HuggingFace repository registered for dataset key '{key}'. "
                f"Available keys: {list(HF_REPO_IDS.keys())}"
            )
        return repo_id


def get_default_device() -> str:
    """Returns 'cuda' if GPU is available and stable, otherwise 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@st.cache_resource(show_spinner="Loading fine-tuned mBERT checkpoint...")
def load_cached_pipeline(dataset_key: str, device: Optional[str] = None) -> Tuple[Any, Any, str]:
    """
    Loads and caches tokenizer and mBERT model for the specified language pair.

    Source is controlled by the MODEL_SOURCE environment variable:
      - 'hf'    (default): downloads from HuggingFace Hub
      - 'local':           loads from local checkpoint directory

    Uses eager attention implementation to support self-attention extraction.
    Falls back gracefully if attn_implementation is unsupported by the installed
    transformers version.

    Args:
        dataset_key: One of 'hecm', 'tamil', 'malayalam'.
        device: Torch device string. Defaults to GPU if available, else CPU.

    Returns:
        Tuple of (model, tokenizer, device_str).
    """
    key = dataset_key.lower()
    model_id = get_model_identifier(key)

    if device is None:
        device = get_default_device()

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Load with attn_implementation='eager' to ensure output_attentions=True
    # functions correctly across transformers versions.
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            attn_implementation="eager",
        )
    except (TypeError, ValueError):
        # Fallback: older transformers versions do not accept attn_implementation
        model = AutoModelForSequenceClassification.from_pretrained(model_id)

    model.to(torch.device(device))
    model.eval()

    return model, tokenizer, device


def predict_sample(
    model: torch.nn.Module,
    tokenizer: Any,
    text: str,
    dataset_key: str,
    device: str = "cpu",
    max_length: int = MAX_SEQ_LENGTH,
) -> Dict[str, Any]:
    """
    Executes model inference on visible text under strict sequence-length limits.

    Returns:
        dict containing:
        - 'predicted_class': int
        - 'confidence': float (0.0 to 1.0)
        - 'probabilities': dict mapping class index (int) to float probability
        - 'visible_text': str of text actually ingested by the model
        - 'token_count': int total content tokens ingested
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Input text must contain non-whitespace characters.")

    visible_text = get_model_visible_text(text, tokenizer, max_length=max_length)
    if not visible_text.strip():
        raise ValueError("Input text produced no visible tokens for model evaluation.")

    inputs = tokenizer(
        visible_text,
        max_length=max_length,
        truncation=True,
        return_tensors="pt"
    )

    token_count = int(inputs["input_ids"].shape[1])
    target_device = torch.device(device)
    inputs = {k: v.to(target_device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

    pred_cls = int(np.argmax(probs))
    conf = float(probs[pred_cls])

    label_mapping = get_labels_for_dataset(dataset_key)
    prob_dict = {i: float(probs[i]) for i in range(len(probs))}

    return {
        "predicted_class": pred_cls,
        "confidence": conf,
        "probabilities": prob_dict,
        "label_mapping": label_mapping,
        "visible_text": visible_text,
        "token_count": token_count,
        "raw_probs": probs,
    }


def get_checkpoint_storage_info() -> Dict[str, Any]:
    """
    Computes storage footprint of the three authoritative local checkpoints.

    Returns a dict with per-checkpoint sizes and totals.
    If local paths are absent (cloud deployment via HF Hub), sizes are reported
    as 0 and the 'exists' flag is False.
    """
    root = Path(PROJECT_ROOT)
    info = {}
    total_bytes = 0
    for key, rel_path in CHECKPOINT_PATHS.items():
        ckpt_path = root / rel_path
        if ckpt_path.exists():
            size = sum(f.stat().st_size for f in ckpt_path.glob("*") if f.is_file())
            info[key] = {
                "path": str(rel_path),
                "size_mb": round(size / (1024 * 1024), 2),
                "exists": True,
            }
            total_bytes += size
        else:
            info[key] = {
                "path": str(rel_path),
                "size_mb": 0.0,
                "exists": False,
            }

    info["total_size_mb"] = round(total_bytes / (1024 * 1024), 2)
    info["total_size_gb"] = round(total_bytes / (1024 * 1024 * 1024), 2)
    info["model_source"] = get_model_source()
    return info
