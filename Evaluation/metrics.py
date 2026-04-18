"""
Evaluation Metrics
==================
Computes Top-1/Top-5 accuracy, confusion matrix, and per-class accuracy
on a given DataLoader.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
from typing import Tuple, Dict
from sklearn.metrics import confusion_matrix


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: str = "cuda",
    num_classes: int = 10,
) -> Dict:
    """
    Full evaluation pass.

    Returns dict with:
        top1_acc    : float  (%)
        top5_acc    : float  (%)
        all_preds   : np.ndarray (N,)
        all_labels  : np.ndarray (N,)
        per_class   : np.ndarray (num_classes,) per-class accuracy (%)
    """
    model.eval()
    all_preds   = []
    all_labels  = []
    correct_top5 = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            output = model(images)
            logits = output[0] if isinstance(output, (tuple, list)) else output

        # Top-1
        _, pred_top1 = logits.topk(1, dim=1)
        all_preds.extend(pred_top1.squeeze(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # Top-5
        k = min(5, logits.size(1))
        _, pred_topk = logits.topk(k, dim=1)
        correct_top5 += pred_topk.eq(labels.view(-1, 1).expand_as(pred_topk)).sum().item()
        total += labels.size(0)

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    top1_acc = 100.0 * (all_preds == all_labels).mean()
    top5_acc = 100.0 * correct_top5 / total

    # Per-class accuracy
    per_class = np.zeros(num_classes)
    for c in range(num_classes):
        mask = all_labels == c
        if mask.sum() > 0:
            per_class[c] = 100.0 * (all_preds[mask] == all_labels[mask]).mean()

    return {
        "top1_acc":   top1_acc,
        "top5_acc":   top5_acc,
        "all_preds":  all_preds,
        "all_labels": all_labels,
        "per_class":  per_class,
    }


def compute_confusion_matrix(all_preds: np.ndarray, all_labels: np.ndarray) -> np.ndarray:
    """Return normalized confusion matrix (row = true, col = predicted)."""
    cm = confusion_matrix(all_labels, all_preds)
    return cm.astype(float) / cm.sum(axis=1, keepdims=True)
