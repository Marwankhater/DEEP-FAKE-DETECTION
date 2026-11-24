"""Frame-level and video-level scores."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def video_scores(frames: pd.DataFrame) -> pd.DataFrame:
    """Average the per-frame fake probability of each video.

    Expects columns video, label and prob. The mean is a deliberately plain rule; it is
    robust to a few bad crops and easy to explain.
    """
    extra = {c: (c, "first") for c in ("method", "dataset") if c in frames.columns}
    return (
        frames.groupby("video")
        .agg(label=("label", "first"), prob=("prob", "mean"), faces=("prob", "size"), **extra)
        .reset_index()
    )


def score(labels, probs, threshold: float = 0.5) -> dict:
    y = np.asarray(labels).astype(int)
    p = np.clip(np.asarray(probs, dtype=float), 1e-7, 1 - 1e-7)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    both_classes = len(np.unique(y)) == 2
    return {
        "count": int(len(y)),
        "real": int((y == 0).sum()),
        "fake": int((y == 1).sum()),
        "auc": float(roc_auc_score(y, p)) if both_classes else float("nan"),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)) if both_classes else float("nan"),
        "precision_fake": float(precision_score(y, pred, zero_division=0)),
        "recall_fake": float(recall_score(y, pred, zero_division=0)),
        "f1_fake": float(f1_score(y, pred, zero_division=0)),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "threshold": threshold,
    }


def best_threshold(labels, probs) -> float:
    """Threshold that maximises balanced accuracy. Chosen on validation data only."""
    y = np.asarray(labels).astype(int)
    p = np.asarray(probs, dtype=float)
    candidates = np.unique(np.round(p, 3))
    if len(candidates) == 0 or len(np.unique(y)) < 2:
        return 0.5
    results = [(balanced_accuracy_score(y, (p >= t).astype(int)), t) for t in candidates]
    return float(max(results)[1])


def by_method(videos: pd.DataFrame) -> dict[str, dict]:
    """Score each manipulation method on its own: all real videos against that method's fakes."""
    if "method" not in videos.columns:
        return {}
    real = videos[videos["label"] == 0]
    out = {}
    for method in sorted(videos.loc[videos["label"] == 1, "method"].unique()):
        part = pd.concat([real, videos[(videos["label"] == 1) & (videos["method"] == method)]])
        out[method] = score(part["label"], part["prob"])
    return out
