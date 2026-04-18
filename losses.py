"""
Loss Functions
==============
LabelSmoothingCrossEntropy: Applies label smoothing to hard targets.
SoftTargetCrossEntropy: Accepts pre-computed soft target distributions
                        (required when using Mixup/CutMix).

Label smoothing (Szegedy et al., 2015) improves calibration by preventing
the model from becoming overconfident. The ViT paper uses smoothing=0.1.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy with label smoothing for hard integer targets.

    Args:
        smoothing (float): Label smoothing factor ε (0 = standard CE, 0.1 = ViT default).
    """

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        assert 0.0 <= smoothing < 1.0
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits  : (B, C) raw logits
            targets : (B,) integer class indices

        Returns:
            Scalar loss value.
        """
        log_probs = F.log_softmax(logits, dim=-1)
        num_classes = logits.size(-1)

        # Construct smoothed target distribution
        # True class gets (1 - ε), all others share ε / (C-1)
        with torch.no_grad():
            smooth_targets = torch.full_like(log_probs, self.smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        loss = -(smooth_targets * log_probs).sum(dim=-1).mean()
        return loss


class SoftTargetCrossEntropy(nn.Module):
    """
    Cross-entropy loss that accepts soft (mixed) target distributions.
    Required when using Mixup or CutMix.

    Args:
        None
    """

    def forward(self, logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits       : (B, C) raw logits
            soft_targets : (B, C) soft probability targets (sum to 1 per sample)

        Returns:
            Scalar loss value.
        """
        log_probs = F.log_softmax(logits, dim=-1)
        return -(soft_targets * log_probs).sum(dim=-1).mean()
