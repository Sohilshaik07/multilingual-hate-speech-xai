"""
src/evaluation/metrics.py

Reusable evaluation metrics and inference functions for transformer models.

Supported Metrics:
- Accuracy
- Macro-averaged F1 score
- Weighted-averaged F1 score
- Per-class Precision, Recall, F1, and Support
- Confusion Matrix

Inference Functions:
- Batched probability inference (predict_proba)
- Prediction generation (predict)

Strict Research Integrity:
- NEVER overwrites authoritative results/final/master_experimental_results.csv
- Verification outputs are saved separately under results/verification/ if requested.
"""

from typing import List, Dict, Any, Optional, Union
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix
)


def compute_accuracy(y_true: Union[List, np.ndarray], y_pred: Union[List, np.ndarray]) -> float:
    """Calculates overall classification accuracy."""
    return float(accuracy_score(y_true, y_pred))


def compute_macro_f1(y_true: Union[List, np.ndarray], y_pred: Union[List, np.ndarray]) -> float:
    """Calculates unweighted macro-averaged F1 score across all classes."""
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def compute_weighted_f1(y_true: Union[List, np.ndarray], y_pred: Union[List, np.ndarray]) -> float:
    """Calculates support-weighted F1 score across all classes."""
    return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))


def compute_per_class_metrics(
    y_true: Union[List, np.ndarray],
    y_pred: Union[List, np.ndarray],
    labels: Optional[List[Any]] = None
) -> pd.DataFrame:
    """
    Computes fine-grained precision, recall, F1-score, and support for each class.
    Returns a DataFrame with columns: ['class', 'precision', 'recall', 'f1', 'support'].
    """
    if labels is None:
        labels = sorted(list(set(y_true) | set(y_pred)))
        
    prec, rec, f1, supp = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    
    return pd.DataFrame({
        "class": labels,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "support": supp
    })


def compute_confusion_matrix(
    y_true: Union[List, np.ndarray],
    y_pred: Union[List, np.ndarray],
    labels: Optional[List[Any]] = None
) -> np.ndarray:
    """Computes confusion matrix where rows represent true labels and columns predicted labels."""
    if labels is None:
        labels = sorted(list(set(y_true) | set(y_pred)))
    return confusion_matrix(y_true, y_pred, labels=labels)


def compute_all_metrics(
    y_true: Union[List, np.ndarray],
    y_pred: Union[List, np.ndarray],
    labels: Optional[List[Any]] = None
) -> Dict[str, Any]:
    """Computes accuracy, macro_f1, weighted_f1, and per_class breakdown in a single dictionary."""
    acc = compute_accuracy(y_true, y_pred)
    macro = compute_macro_f1(y_true, y_pred)
    weighted = compute_weighted_f1(y_true, y_pred)
    per_class_df = compute_per_class_metrics(y_true, y_pred, labels=labels)
    cm = compute_confusion_matrix(y_true, y_pred, labels=labels)
    
    return {
        "accuracy": acc,
        "macro_f1": macro,
        "weighted_f1": weighted,
        "per_class": per_class_df.to_dict(orient="records"),
        "confusion_matrix": cm.tolist()
    }


def predict_proba(
    model: torch.nn.Module,
    tokenizer: Any,
    texts: List[str],
    device: Optional[str] = None,
    batch_size: int = 32,
    max_length: int = 128
) -> np.ndarray:
    """
    Runs batched inference and returns softmax predicted probabilities.
    Output shape: (len(texts), num_classes).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    target_device = torch.device(device)
    model.to(target_device)
    model.eval()

    all_probs = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = [str(t) if t is not None else "" for t in texts[i:i + batch_size]]
        encodings = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
        input_ids = encodings["input_ids"].to(target_device)
        attention_mask = encodings["attention_mask"].to(target_device)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
            
    if all_probs:
        return np.vstack(all_probs)
    return np.empty((0, model.config.num_labels if hasattr(model, "config") else 0))


def predict(
    model: torch.nn.Module,
    tokenizer: Any,
    texts: List[str],
    device: Optional[str] = None,
    batch_size: int = 32,
    max_length: int = 128
) -> np.ndarray:
    """
    Runs batched inference and returns argmax class prediction indices.
    Output shape: (len(texts),).
    """
    probs = predict_proba(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        device=device,
        batch_size=batch_size,
        max_length=max_length
    )
    if len(probs) == 0:
        return np.array([])
    return np.argmax(probs, axis=1)
