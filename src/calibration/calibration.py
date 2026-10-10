"""
src/calibration/calibration.py

Authoritative calibration evaluation module implementing:
- Expected Calibration Error (ECE) with 15 uniform bins
- Multiclass Brier Score: (1/N) * sum_i sum_k (p_ik - y_ik)^2
- Reliability binning matching the authoritative schema
- Confidence and accuracy gap analysis
"""

from typing import Tuple, Dict, Any, Optional, Union, List
import numpy as np
import pandas as pd


def extract_confidences(probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts the winning class confidence (maximum predicted probability)
    and predicted class index for each sample.
    
    Args:
        probs: Array of softmax probabilities of shape (N, num_classes).
        
    Returns:
        confidences: Array of shape (N,) containing max probabilities.
        predictions: Array of shape (N,) containing argmax class indices.
    """
    probs = np.asarray(probs)
    if probs.ndim != 2:
        raise ValueError(f"Expected 2D probability array of shape (N, num_classes), got shape {probs.shape}")
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    return confidences, predictions


def compute_brier_score(
    probs: np.ndarray,
    y_true: Union[List[int], np.ndarray],
    num_classes: Optional[int] = None
) -> float:
    """
    Computes the multiclass Brier score:
        Brier = (1 / N) * sum_{i=1}^N sum_{k=1}^K (p_{ik} - y_{ik})^2
    where y_{ik} is 1 if true label is k, and 0 otherwise.
    
    This matches the authoritative formulation used across HECM, Tamil, and Malayalam.
    """
    probs = np.asarray(probs, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int64)
    n_samples = len(y_true)
    
    if n_samples == 0:
        return 0.0
        
    if num_classes is None:
        num_classes = probs.shape[1]
        
    # One-hot encode ground truth
    one_hot = np.zeros((n_samples, num_classes), dtype=np.float64)
    for i, label in enumerate(y_true):
        if 0 <= label < num_classes:
            one_hot[i, label] = 1.0
        else:
            raise ValueError(f"Label {label} at index {i} out of bounds for num_classes={num_classes}")
            
    squared_diff = (probs - one_hot) ** 2
    brier = float(np.sum(squared_diff) / n_samples)
    return brier


def compute_reliability_bins(
    y_true: Union[List[int], np.ndarray],
    probs: np.ndarray,
    num_bins: int = 15
) -> pd.DataFrame:
    """
    Partitions confidence scores into uniform bins in [0, 1] and computes:
    - bin: 1-indexed bin integer
    - lower: lower probability threshold
    - upper: upper probability threshold
    - count: number of predictions in this bin
    - mean_confidence: average confidence of predictions in this bin
    - accuracy: empirical accuracy of predictions in this bin
    - gap: |mean_confidence - accuracy|
    
    Matches the authoritative schema of hecm_reliability_bins.csv.
    Empty bins have count=0 and NaN for mean_confidence, accuracy, and gap.
    """
    probs = np.asarray(probs)
    y_true = np.asarray(y_true, dtype=np.int64)
    confidences, predictions = extract_confidences(probs)
    correct = (predictions == y_true).astype(float)
    
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    records = []
    
    for b in range(num_bins):
        lower = bin_edges[b]
        upper = bin_edges[b + 1]
        
        # Include upper boundary in the last bin
        if b == num_bins - 1:
            in_bin = (confidences >= lower) & (confidences <= upper)
        else:
            in_bin = (confidences >= lower) & (confidences < upper)
            
        count = int(np.sum(in_bin))
        if count > 0:
            mean_conf = float(np.mean(confidences[in_bin]))
            acc = float(np.mean(correct[in_bin]))
            gap = float(abs(mean_conf - acc))
        else:
            mean_conf = np.nan
            acc = np.nan
            gap = np.nan
            
        records.append({
            "bin": b + 1,
            "lower": float(lower),
            "upper": float(upper),
            "count": count,
            "mean_confidence": mean_conf,
            "accuracy": acc,
            "gap": gap
        })
        
    return pd.DataFrame(records)


def compute_ece_from_bins(bins_df: pd.DataFrame) -> float:
    """
    Computes Expected Calibration Error (ECE) directly from a reliability bins DataFrame:
        ECE = sum_{b: count_b > 0} (count_b / total_count) * gap_b
    """
    total_count = bins_df["count"].sum()
    if total_count == 0:
        return 0.0
    valid_bins = bins_df[bins_df["count"] > 0]
    weighted_gap = (valid_bins["count"] * valid_bins["gap"]).sum()
    return float(weighted_gap / total_count)


def compute_calibration_summary(
    y_true: Union[List[int], np.ndarray],
    probs: np.ndarray,
    num_bins: int = 15
) -> Dict[str, Any]:
    """
    Computes a complete calibration summary dictionary:
    - ECE
    - Brier Score
    - Mean Confidence
    - Empirical Accuracy
    - Reliability Bins DataFrame
    """
    probs = np.asarray(probs)
    y_true = np.asarray(y_true, dtype=np.int64)
    confidences, predictions = extract_confidences(probs)
    
    bins_df = compute_reliability_bins(y_true, probs, num_bins=num_bins)
    ece = compute_ece_from_bins(bins_df)
    brier = compute_brier_score(probs, y_true)
    mean_conf = float(np.mean(confidences))
    acc = float(np.mean(predictions == y_true))
    
    return {
        "ece": ece,
        "brier_score": brier,
        "mean_confidence": mean_conf,
        "accuracy": acc,
        "num_bins": num_bins,
        "total_samples": len(y_true),
        "bins_df": bins_df
    }
