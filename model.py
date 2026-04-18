"""
Vision Transformer (ViT) — Full Model
=======================================
Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for
Image Recognition at Scale", ICLR 2021. arXiv:2010.11929

Architecture summary:
    1. Patch Embedding  : image -> sequence of patch tokens + CLS token
    2. Transformer Enc  : depth × [MHSA + MLP] with Pre-LN and residuals
    3. Classification   : LayerNorm(CLS token) -> Linear(D, num_classes)

The CLS token (index 0 of the sequence) aggregates global information across
all patches through self-attention and is used as the image-level representation
for classification. This is analogous to the [CLS] token in BERT.
"""

import torch
import torch.nn as nn
from timm.layers import trunc_normal_
from typing import List, Optional, Dict, Any

from .components.patch_embedding import PatchEmbedding
from .components.encoder_block import TransformerEncoderBlock
from .configs import VIT_CONFIGS


class VisionTransformer(nn.Module):
    """
    Args:
        img_size       (int): Square input image dimension.
        patch_size     (int): Square patch dimension.
        in_channels    (int): Input channels (3 for RGB).
        num_classes    (int): Number of output classes.
        embed_dim      (int): Token embedding dimension D.
        depth          (int): Number of transformer encoder blocks.
        num_heads      (int): Number of attention heads per block.
        mlp_ratio      (float): MLP hidden dimension ratio.
        dropout        (float): General dropout (patch embed output, MLP, proj).
        attn_dropout   (float): Attention weight dropout.
        drop_path_rate (float): Max stochastic depth rate (linearly scheduled per layer).
        global_avg_pool(bool): Use GAP over all patch tokens instead of CLS token.
                               The paper uses CLS (False); some follow-up work prefers GAP.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        drop_path_rate: float = 0.1,
        global_avg_pool: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        self.global_avg_pool = global_avg_pool

        # ── 1. Patch Embedding ───────────────────────────────────────
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            dropout=dropout,
        )
        num_patches = self.patch_embed.num_patches

        # ── 2. Transformer Encoder ───────────────────────────────────
        # Linearly increase drop_path_rate per block (stochastic depth schedule)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                attn_drop=attn_dropout,
                proj_drop=dropout,
                drop_path_rate=dpr[i],
            )
            for i in range(depth)
        ])

        # ── 3. Classification Head ───────────────────────────────────
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        """Initialize all weights following the JAX ViT reference."""
        def _init(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.apply(_init)

        # Zero-init the classification head (better calibration)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        x: torch.Tensor,
        return_attn_weights: bool = False,
    ):
        """
        Args:
            x                  : (B, C, H, W)
            return_attn_weights: if True, return list of attention maps from all blocks

        Returns:
            logits      : (B, num_classes)
            attn_list   : list of (B, h, N+1, N+1) tensors per block, or None
        """
        # Patch embedding: (B, C, H, W) -> (B, N+1, D)
        x = self.patch_embed(x)

        attn_list = [] if return_attn_weights else None

        # Transformer encoder
        for block in self.blocks:
            x, attn = block(x, return_attn_weights=return_attn_weights)
            if return_attn_weights:
                attn_list.append(attn)

        # Final normalization
        x = self.norm(x)

        # Aggregate to single vector
        if self.global_avg_pool:
            # Average over all patch tokens (skip CLS at index 0)
            cls_repr = x[:, 1:, :].mean(dim=1)
        else:
            # Use CLS token only
            cls_repr = x[:, 0, :]

        logits = self.head(cls_repr)

        return logits, attn_list

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def no_weight_decay_params(self) -> List[str]:
        """
        Return parameter names that should NOT have weight decay applied.
        Standard practice: exclude all 1D parameters (biases, LayerNorm, pos_embed, cls_token).
        """
        no_wd = set()
        for name, param in self.named_parameters():
            if param.ndim == 1 or "pos_embed" in name or "cls_token" in name:
                no_wd.add(name)
        return list(no_wd)


def build_vit(
    config_name: str,
    num_classes: int = 10,
    **kwargs,
) -> VisionTransformer:
    """
    Build a ViT model from a named configuration.

    Args:
        config_name: Key in VIT_CONFIGS (e.g. 'vit_tiny_patch4_32').
        num_classes: Number of output classes.
        **kwargs   : Override any config parameter.

    Returns:
        VisionTransformer instance.
    """
    if config_name not in VIT_CONFIGS:
        raise ValueError(
            f"Unknown config '{config_name}'. "
            f"Available: {list(VIT_CONFIGS.keys())}"
        )
    cfg: Dict[str, Any] = {**VIT_CONFIGS[config_name], **kwargs}
    cfg["num_classes"] = num_classes
    return VisionTransformer(**cfg)
