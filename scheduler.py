"""
Learning Rate Scheduler
========================
Implements the warmup + cosine annealing schedule used in the ViT paper.

Schedule:
  Phase 1 (Warmup): linearly increase lr from 0 to base_lr over warmup_epochs.
  Phase 2 (Cosine): cosine decay from base_lr to min_lr over remaining epochs.

This is critical for ViT training stability. Without warmup, the attention
mechanism can diverge in early training due to random weight initialization.
"""

import torch
import math
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineScheduler(_LRScheduler):
    """
    Warmup + Cosine Annealing learning rate schedule.

    Args:
        optimizer      : PyTorch optimizer.
        warmup_epochs  : Number of epochs for linear warmup.
        total_epochs   : Total training epochs.
        min_lr         : Minimum LR at end of cosine decay (default 1e-6).
        last_epoch     : Used for resuming training. Default: -1 (start fresh).
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 1e-6,
        last_epoch: int = -1,
    ):
        self.warmup_epochs = warmup_epochs
        self.total_epochs  = total_epochs
        self.min_lr        = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch

        if epoch < self.warmup_epochs:
            # Linear warmup: lr_scale = epoch / warmup_epochs
            scale = (epoch + 1) / max(self.warmup_epochs, 1)
        else:
            # Cosine annealing from base_lr to min_lr
            progress = (epoch - self.warmup_epochs) / max(
                self.total_epochs - self.warmup_epochs, 1
            )
            scale = 0.5 * (1 + math.cos(math.pi * progress))

        return [
            self.min_lr + (base_lr - self.min_lr) * scale
            for base_lr in self.base_lrs
        ]


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int = 10,
    total_epochs: int = 300,
    min_lr: float = 1e-6,
) -> WarmupCosineScheduler:
    return WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_epochs=warmup_epochs,
        total_epochs=total_epochs,
        min_lr=min_lr,
    )
