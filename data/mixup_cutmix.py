"""
Mixup and CutMix Augmentation
==============================
Both techniques operate in the batch dimension (applied after loading a batch)
and produce soft labels, requiring LabelSmoothingCrossEntropy or soft-target CE.

Mixup  (Zhang et al., 2018): convex combination of two images and their labels.
CutMix (Yun et al., 2019) : replace a random rectangular region in one image
                              with the corresponding region from another image.

Usage in training loop:
    mixup_fn = MixupCutmix(mixup_alpha=0.2, cutmix_alpha=1.0, num_classes=10)
    images, labels = mixup_fn(images, labels)
    # labels are now soft one-hot vectors
"""

import torch
import numpy as np
from typing import Tuple


def mixup_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Return mixed inputs, pairs of targets, and lambda."""
    if alpha > 0:
        lam = float(np.random.beta(alpha, alpha))
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Return cutmix images, pairs of targets, and lambda."""
    if alpha > 0:
        lam = float(np.random.beta(alpha, alpha))
    else:
        lam = 1.0

    batch_size, _, H, W = x.shape
    index = torch.randperm(batch_size, device=x.device)

    # Sample cut box
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    mixed_x = x.clone()
    mixed_x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]

    # Adjust lambda to be proportional to actual replaced area
    lam = 1 - (x2 - x1) * (y2 - y1) / (W * H)
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


class MixupCutmix:
    """
    Randomly applies either Mixup or CutMix to a batch with equal probability.
    Falls back to plain batch when both alphas are 0.

    Args:
        mixup_alpha  (float): Beta distribution parameter for Mixup.
        cutmix_alpha (float): Beta distribution parameter for CutMix.
        num_classes  (int): Number of classes for one-hot encoding.
        switch_prob  (float): Probability of using CutMix vs Mixup.
    """

    def __init__(
        self,
        mixup_alpha: float = 0.2,
        cutmix_alpha: float = 1.0,
        num_classes: int = 10,
        switch_prob: float = 0.5,
    ):
        self.mixup_alpha  = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.num_classes  = num_classes
        self.switch_prob  = switch_prob

    def __call__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, C, H, W) image batch
            y: (B,) integer class labels

        Returns:
            x_mixed : (B, C, H, W)
            y_soft  : (B, num_classes) soft label tensor
        """
        use_cutmix = (self.cutmix_alpha > 0) and (np.random.random() < self.switch_prob)

        if use_cutmix:
            x, y_a, y_b, lam = cutmix_data(x, y, self.cutmix_alpha)
        else:
            x, y_a, y_b, lam = mixup_data(x, y, self.mixup_alpha)

        # Convert to soft one-hot targets
        y_a_one_hot = torch.zeros(x.size(0), self.num_classes, device=x.device)
        y_b_one_hot = torch.zeros(x.size(0), self.num_classes, device=x.device)
        y_a_one_hot.scatter_(1, y_a.unsqueeze(1), 1)
        y_b_one_hot.scatter_(1, y_b.unsqueeze(1), 1)

        y_soft = lam * y_a_one_hot + (1 - lam) * y_b_one_hot
        return x, y_soft
