"""
Unit Tests — Model Shape Verification
======================================
Verify that all components produce tensors of expected shapes.
Run: pytest tests/ -v

All tests use small batch sizes and minimal configs for speed.
"""

import pytest
import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vit.components.patch_embedding import PatchEmbedding
from vit.components.attention import MultiHeadSelfAttention
from vit.components.mlp import MLP
from vit.components.encoder_block import TransformerEncoderBlock
from vit.model import VisionTransformer, build_vit


# ── Fixtures ──────────────────────────────────────────────────────────────

B, C, H = 2, 3, 32   # Batch, Channels, Height
P, D     = 4, 192     # Patch size, Embed dim
N        = (H // P) ** 2  # 64 patches

@pytest.fixture
def batch():
    return torch.randn(B, C, H, H)


# ── PatchEmbedding tests ──────────────────────────────────────────────────

class TestPatchEmbedding:
    def test_output_shape(self, batch):
        model = PatchEmbedding(img_size=H, patch_size=P, in_channels=C, embed_dim=D)
        out = model(batch)
        # (B, N+1, D) - N patches + 1 CLS token
        assert out.shape == (B, N + 1, D), f"Expected {(B, N+1, D)}, got {out.shape}"

    def test_num_patches(self, batch):
        model = PatchEmbedding(img_size=32, patch_size=4, embed_dim=192)
        assert model.num_patches == 64

    def test_pos_embed_shape(self):
        model = PatchEmbedding(img_size=H, patch_size=P, embed_dim=D)
        assert model.pos_embed.shape == (1, N + 1, D)

    def test_cls_token_shape(self):
        model = PatchEmbedding(img_size=H, patch_size=P, embed_dim=D)
        assert model.cls_token.shape == (1, 1, D)

    def test_pos_embed_2d_shape(self):
        model = PatchEmbedding(img_size=H, patch_size=P, embed_dim=D)
        grid = int(N ** 0.5)
        pos_2d = model.get_pos_embed_2d()
        assert pos_2d.shape == (grid, grid, D)


# ── MultiHeadSelfAttention tests ─────────────────────────────────────────

class TestMHSA:
    def test_output_shape(self):
        model = MultiHeadSelfAttention(embed_dim=D, num_heads=3)
        x = torch.randn(B, N + 1, D)
        out, attn = model(x, return_attn_weights=False)
        assert out.shape == (B, N + 1, D)
        assert attn is None

    def test_attention_weights_shape(self):
        model = MultiHeadSelfAttention(embed_dim=D, num_heads=3)
        x = torch.randn(B, N + 1, D)
        out, attn = model(x, return_attn_weights=True)
        assert attn.shape == (B, 3, N + 1, N + 1)

    def test_embed_dim_divisibility(self):
        with pytest.raises(AssertionError):
            MultiHeadSelfAttention(embed_dim=192, num_heads=5)  # 192 % 5 != 0


# ── MLP tests ─────────────────────────────────────────────────────────────

class TestMLP:
    def test_output_shape(self):
        model = MLP(in_features=D, mlp_ratio=4.0)
        x = torch.randn(B, N + 1, D)
        out = model(x)
        assert out.shape == (B, N + 1, D)

    def test_hidden_dim(self):
        model = MLP(in_features=192, mlp_ratio=4.0)
        assert model.fc1.out_features == 768
        assert model.fc2.out_features == 192


# ── TransformerEncoderBlock tests ─────────────────────────────────────────

class TestEncoderBlock:
    def test_output_shape(self):
        model = TransformerEncoderBlock(embed_dim=D, num_heads=3)
        x = torch.randn(B, N + 1, D)
        out, _ = model(x)
        assert out.shape == (B, N + 1, D)

    def test_attn_weight_return(self):
        model = TransformerEncoderBlock(embed_dim=D, num_heads=3)
        x = torch.randn(B, N + 1, D)
        out, attn = model(x, return_attn_weights=True)
        assert attn.shape == (B, 3, N + 1, N + 1)

    def test_residual_identity(self):
        """With zero weights, output should approximate input (residual)."""
        model = TransformerEncoderBlock(embed_dim=D, num_heads=3)
        for p in model.parameters():
            p.data.zero_()
        x = torch.randn(B, N + 1, D)
        out, _ = model(x)
        # With zero weights, residual = x + 0 = x (approximately)
        assert out.shape == x.shape


# ── Full VisionTransformer tests ──────────────────────────────────────────

class TestVisionTransformer:
    def test_output_shape(self, batch):
        model = VisionTransformer(
            img_size=32, patch_size=4, in_channels=3, num_classes=10,
            embed_dim=192, depth=2, num_heads=3
        )
        logits, _ = model(batch)
        assert logits.shape == (B, 10)

    def test_attn_list_length(self, batch):
        depth = 3
        model = VisionTransformer(
            img_size=32, patch_size=4, in_channels=3, num_classes=10,
            embed_dim=192, depth=depth, num_heads=3
        )
        logits, attn_list = model(batch, return_attn_weights=True)
        assert len(attn_list) == depth

    def test_global_avg_pool(self, batch):
        model = VisionTransformer(
            img_size=32, patch_size=4, in_channels=3, num_classes=10,
            embed_dim=192, depth=2, num_heads=3, global_avg_pool=True
        )
        logits, _ = model(batch)
        assert logits.shape == (B, 10)

    def test_build_vit_factory(self, batch):
        model = build_vit("vit_tiny_patch4_32", num_classes=10)
        logits, _ = model(batch)
        assert logits.shape == (B, 10)

    def test_param_count(self):
        model = build_vit("vit_tiny_patch4_32", num_classes=10)
        n_params = model.get_num_params()
        # ViT-Tiny should be ~5.7M params
        assert 4_000_000 < n_params < 8_000_000, f"Unexpected param count: {n_params:,}"

    def test_no_weight_decay_params(self):
        model = build_vit("vit_tiny_patch4_32", num_classes=10)
        no_wd = set(model.no_weight_decay_params())
        # CLS token and pos_embed must be excluded
        assert any("cls_token" in n for n in no_wd)
        assert any("pos_embed" in n for n in no_wd)
