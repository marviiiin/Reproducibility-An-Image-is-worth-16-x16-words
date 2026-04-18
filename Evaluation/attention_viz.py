"""
Attention Rollout Visualization
================================
Implements the attention rollout algorithm from:
  Abnar & Zuidema (2020), "Quantifying Attention Flow in Transformers"
  https://arxiv.org/abs/2005.00928

Algorithm:
  1. For each transformer layer l, compute attention_l with residual connection:
       A_l = 0.5 * attention_l + 0.5 * I   (I = identity, modeling the residual path)
  2. Average over attention heads: A_l = mean over h of A_l^h
  3. Propagate through all layers: rollout = A_L @ A_{L-1} @ ... @ A_1
  4. Extract CLS token row (index 0): rollout[0, 1:]  -> per-patch attribution
  5. Reshape to (grid_size, grid_size) and upsample to image size

The resulting heatmap shows which image regions the CLS token "attends to"
after integrating information across all layers, accounting for residual paths.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from typing import List, Optional, Tuple


class AttentionRollout:
    """
    Computes attention rollout maps for a VisionTransformer.

    Args:
        model  : Trained VisionTransformer instance.
        device : 'cuda' or 'cpu'.
        head_fusion: How to combine attention heads.
                     'mean' (default), 'min', or 'max'.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        head_fusion: str = "mean",
    ):
        self.model = model.to(device)
        self.device = device
        self.head_fusion = head_fusion

    @torch.no_grad()
    def __call__(self, img_tensor: torch.Tensor) -> np.ndarray:
        """
        Compute rollout map for a single image.

        Args:
            img_tensor: (1, C, H, W) preprocessed image tensor.

        Returns:
            mask: (grid_h, grid_w) numpy array, values in [0, 1].
        """
        img_tensor = img_tensor.to(self.device)
        logits, attn_list = self.model(img_tensor, return_attn_weights=True)

        # attn_list: list of (1, h, N+1, N+1) tensors per layer
        rollout = self._compute_rollout(attn_list)

        # CLS token row, skip CLS token itself (index 0) -> patch attributions
        cls_attn = rollout[0, 1:]  # (N,)

        # Reshape to 2D patch grid
        grid_size = int(cls_attn.shape[0] ** 0.5)
        mask = cls_attn.reshape(grid_size, grid_size).numpy()

        # Normalize to [0, 1]
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
        return mask

    def _compute_rollout(self, attn_list: List[torch.Tensor]) -> torch.Tensor:
        """Propagate attention through layers with residual path modelling."""
        result = None

        for attn in attn_list:
            # attn: (B, h, N, N)
            attn = attn.cpu()

            # Fuse heads
            if self.head_fusion == "mean":
                attn_fused = attn.mean(dim=1)   # (B, N, N)
            elif self.head_fusion == "min":
                attn_fused = attn.min(dim=1)[0]
            elif self.head_fusion == "max":
                attn_fused = attn.max(dim=1)[0]

            # Add residual connection (identity matrix scaled by 0.5)
            B, N, _ = attn_fused.shape
            identity = torch.eye(N, device=attn_fused.device).unsqueeze(0).expand(B, -1, -1)
            attn_fused = 0.5 * attn_fused + 0.5 * identity

            # Re-normalize rows to sum to 1
            attn_fused = attn_fused / attn_fused.sum(dim=-1, keepdim=True)

            if result is None:
                result = attn_fused
            else:
                result = torch.bmm(attn_fused, result)

        return result[0]  # (N+1, N+1)


def visualize_attention(
    rollout: AttentionRollout,
    images: torch.Tensor,
    raw_images: List,
    class_names: List[str],
    labels: torch.Tensor,
    save_path: str,
    num_images: int = 8,
):
    """
    Produce a grid of images with attention rollout heatmap overlay.

    Args:
        rollout    : AttentionRollout instance.
        images     : (B, C, H, W) preprocessed batch.
        raw_images : List of PIL images (original, before normalization).
        class_names: List of class name strings.
        labels     : (B,) ground truth labels.
        save_path  : Output PNG path.
        num_images : Number of samples to show.
    """
    n = min(num_images, len(images))
    fig, axes = plt.subplots(2, n, figsize=(n * 3, 6))
    fig.patch.set_facecolor("#0E2841")

    for i in range(n):
        img_t = images[i:i+1]
        mask = rollout(img_t)

        # Upsample mask to raw image size
        raw = raw_images[i]
        if isinstance(raw, torch.Tensor):
            h, w = raw.shape[-2], raw.shape[-1]
        else:
            w, h = raw.size

        mask_up = np.array(
            Image.fromarray((mask * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
        ) / 255.0

        # Show original image
        if isinstance(raw, torch.Tensor):
            img_np = raw.permute(1, 2, 0).cpu().numpy()
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        else:
            img_np = np.array(raw) / 255.0

        axes[0, i].imshow(img_np)
        axes[0, i].set_title(class_names[labels[i]], color="white", fontsize=9)
        axes[0, i].axis("off")

        # Show attention overlay
        axes[1, i].imshow(img_np)
        axes[1, i].imshow(mask_up, cmap="jet", alpha=0.5)
        axes[1, i].set_title("Attention", color="white", fontsize=9)
        axes[1, i].axis("off")

    plt.suptitle("Attention Rollout Maps", color="white", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0E2841")
    plt.close()
    print(f"Attention map saved: {save_path}")
