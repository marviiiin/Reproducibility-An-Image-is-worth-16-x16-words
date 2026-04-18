"""
ViT Model Configuration Registry
==================================
All model variants from the original paper plus a ViT-Tiny for small-scale experiments.

Parameter counts (approx, at img_size=224, patch_size=16):
  ViT-Tiny  :   5.7M
  ViT-Small :  22.0M
  ViT-Base  :  86.6M
  ViT-Large : 307.0M
  ViT-Huge  : 632.0M

For CIFAR-10 (32x32), patch_size=4 is recommended to keep N=64 patches (8x8 grid).
Using patch_size=16 on 32x32 gives only N=4 patches, which is insufficient.
"""

VIT_CONFIGS = {
    # ── Small-scale configs (for CIFAR-10 from scratch) ──────────────────────
    "vit_tiny_patch4_32": dict(
        img_size=32, patch_size=4, in_channels=3,
        embed_dim=192, depth=12, num_heads=3,
        mlp_ratio=4.0, dropout=0.1, attn_dropout=0.0,
        drop_path_rate=0.1,
    ),
    "vit_small_patch4_32": dict(
        img_size=32, patch_size=4, in_channels=3,
        embed_dim=384, depth=12, num_heads=6,
        mlp_ratio=4.0, dropout=0.1, attn_dropout=0.0,
        drop_path_rate=0.1,
    ),

    # ── Standard configs (original paper, for 224x224 fine-tuning) ───────────
    "vit_tiny_patch16_224": dict(
        img_size=224, patch_size=16, in_channels=3,
        embed_dim=192, depth=12, num_heads=3,
        mlp_ratio=4.0, dropout=0.1, attn_dropout=0.0,
        drop_path_rate=0.1,
    ),
    "vit_small_patch16_224": dict(
        img_size=224, patch_size=16, in_channels=3,
        embed_dim=384, depth=12, num_heads=6,
        mlp_ratio=4.0, dropout=0.1, attn_dropout=0.0,
        drop_path_rate=0.1,
    ),
    "vit_base_patch16_224": dict(
        img_size=224, patch_size=16, in_channels=3,
        embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4.0, dropout=0.1, attn_dropout=0.0,
        drop_path_rate=0.1,
    ),
    "vit_large_patch16_224": dict(
        img_size=224, patch_size=16, in_channels=3,
        embed_dim=1024, depth=24, num_heads=16,
        mlp_ratio=4.0, dropout=0.1, attn_dropout=0.0,
        drop_path_rate=0.2,
    ),
    "vit_base_patch32_224": dict(
        img_size=224, patch_size=32, in_channels=3,
        embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4.0, dropout=0.1, attn_dropout=0.0,
        drop_path_rate=0.1,
    ),
}
