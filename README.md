
# Reproducing "An Image is Worth 16×16 Words"
### Vision Transformer (ViT) — PyTorch Implementation on CIFAR-10

> **Paper:** Dosovitskiy et al. (2021), *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*, ICLR 2021. [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)

A complete, from-scratch PyTorch reproduction of the Vision Transformer (ViT), evaluated on CIFAR-10 with 5 systematic experiments. Includes a publishable reproducibility report, a 28-slide PowerPoint presentation, and structural limitations analysis.

---
The model was pretrained on CIFAR - 10 dataset

![image](https://github.com/marviiiin/Reproducibility-An-Image-is-worth-16-x16-words/blob/c229a95518337842aad172c45b2d0c67beeb2562/cifar-10.png)


## Key Results

| Model | Dataset | Test Top-1 | Test Top-5 | Params | GFLOPs | Epochs |
|---|---|---|---|---|---|---|
| ResNet-18 (CNN baseline) | CIFAR-10 | 94.37% | 99.35% | 11.1M | 0.56G | 200 |
| ViT-Tiny/4 (from scratch) | CIFAR-10 | 87.38% | 97.49% | 5.7M | 0.38G | 300 |
| ViT-Tiny/4 (scratch + Aug) | CIFAR-10 | 92.58% | 99.66% | 5.7M | 0.38G | 300 |
| **ViT-Base/16 (pretrained)** | CIFAR-10 | **98.68%** | 99.95% | 86.0M | 17.6G | 30 |
| ViT-Base/16 (original paper) | CIFAR-10 | 98.13% | — | 86.0M | 17.6G | — |

**Central Claim Confirmed:** Fine-tuned ViT-Base surpasses all from-scratch models (+11.3% vs scratch ViT-Tiny, +4.3% vs ResNet-18), validating the paper's core finding that scale and pre-training are decisive. Our fine-tuned result (98.68%) marginally exceeds the original paper's reported 98.13%.

---

## Architecture Overview
![image](https://github.com/marviiiin/Reproducibility-An-Image-is-worth-16-x16-words/blob/443181da50d96853210f51759f8fc0d61639866b/vit_figure.png)
```
Image (B, 3, 32, 32)
       ↓
Patch Embedding (Conv2d P=4, stride=4)
       ↓
[CLS] token prepended → (B, 65, 192)
       ↓
+ Positional Embeddings (learnable 1D)
       ↓
┌─────────────────────────────────┐
│  Transformer Encoder Block ×12  │
│  ┌──────────────────────────┐   │
│  │ LayerNorm                │   │
│  │ Multi-Head Self-Attention│   │
│  │ + Residual               │   │
│  │ LayerNorm                │   │
│  │ MLP (GELU, ratio=4)      │   │
│  │ + Residual               │   │
│  └──────────────────────────┘   │
└─────────────────────────────────┘
       ↓
LayerNorm → CLS token → Linear(192, 10)
       ↓
Class Logits (B, 10)
```

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/marviiiin/vit-reproducibility
cd vit-reproducibility
pip install -r requirements.txt

# 2. Run unit tests (verify installation)
pytest tests/ -v

# 3. Experiment 1: ViT-Tiny from scratch
python scripts/train.py --config configs/vit_tiny_cifar10.yaml

# 4. Experiment 2: ViT-Tiny with strong augmentation
python scripts/train.py --config configs/vit_tiny_augmented_cifar10.yaml

# 5. Experiment 3: Fine-tune pretrained ViT-Base (recommended first run)
python scripts/train.py --config configs/vit_base_finetune_cifar10.yaml

# 6. ResNet-18 baseline
python scripts/train.py --config configs/baseline_resnet.yaml

# 7. Evaluate a checkpoint
python scripts/evaluate.py \
    --config configs/vit_tiny_cifar10.yaml \
    --checkpoint checkpoints/vit_tiny_scratch_best.pth

# 8. Visualize attention maps
python scripts/visualize_attention.py \
    --config configs/vit_base_finetune_cifar10.yaml \
    --checkpoint checkpoints/vit_base_finetune_best.pth

# 9. Generate all report figures
python scripts/generate_report_figures.py

# 10. Build the Word report
python report/md_to_docx.py

# 11. Build the PowerPoint presentation
python presentation/generate_pptx.py
```

---

## Project Structure

```
reproducibility/
├── vit/                          # Core ViT implementation
│   ├── model.py                  # VisionTransformer + build_vit()
│   ├── configs.py                # Model variant configs
│   └── components/
│       ├── patch_embedding.py    # PatchEmbedding (Conv2d + CLS + PosEmbed)
│       ├── attention.py          # Multi-Head Self-Attention (fused QKV)
│       ├── mlp.py                # Feed-Forward MLP (GELU)
│       └── encoder_block.py      # TransformerEncoderBlock (Pre-LN)
├── data/
│   ├── cifar10_loader.py         # CIFAR-10 dataloaders
│   ├── transforms.py             # Train/val augmentation pipelines
│   └── mixup_cutmix.py           # Mixup + CutMix implementation
├── training/
│   ├── trainer.py                # Main training loop (AMP + TensorBoard)
│   ├── scheduler.py              # Warmup + Cosine LR schedule
│   └── losses.py                 # LabelSmoothing + SoftTarget CE
├── evaluation/
│   ├── metrics.py                # Top-1/5, confusion matrix, per-class acc
│   └── attention_viz.py          # Attention rollout (Abnar & Zuidema 2020)
├── scripts/
│   ├── train.py                  # Training entry point
│   ├── evaluate.py               # Test set evaluation
│   ├── visualize_attention.py    # Attention map generation
│   └── generate_report_figures.py# All matplotlib figures for report
├── configs/                      # YAML experiment configurations

```

---

## Implementation Details

### Key Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| Patch projection | `Conv2d(3, D, kernel=P, stride=P)` | Equivalent to flatten+linear but uses cuDNN, faster |
| Normalization | Pre-LN (LayerNorm before attention) | More stable gradients; used in final JAX ViT release |
| QKV projection | Single fused `Linear(D, 3D)` split | More memory-efficient than 3 separate projections |
| Positional encoding | Learnable 1D | Paper's Appendix D.4: matches or beats fixed 2D sinusoidal |
| Weight init | `trunc_normal_(std=0.02)` | Matches JAX reference implementation |
| Weight decay | Excluded for 1D params, pos_embed, cls_token | Standard ViT practice; significantly affects accuracy |
| Stochastic depth | DropPath, linearly increasing rate | Regularization for deep ViT training |
| Mixed precision | `torch.cuda.amp` | ~2× speedup with minimal accuracy impact |

### CIFAR-10 Adaptation

The original ViT uses 224×224 images with 16×16 patches (196 tokens). For CIFAR-10's 32×32 images:
- **patch_size=4** → 8×8 grid → **64 patches** (comparable to original)
- **patch_size=16** → 2×2 grid → **4 patches** (too few, unusable)
- For fine-tuning Experiment 3: resize CIFAR-10 to 224×224 and use ImageNet normalization stats

---

## Reproducing Paper Claims

| Paper Claim | Our Result | Status |
|---|---|---|
| ViT needs large-scale pretraining to match CNNs | ViT-Tiny scratch 87.2% vs ResNet-18 93.5% | ✓ CONFIRMED |
| Pretrained ViT far outperforms from-scratch | ViT-Base fine-tuned 98.5% vs scratch 87.2% | ✓ CONFIRMED |
| Strong augmentation helps ViT on small datasets | +2.9% with RandAugment+Mixup+CutMix | ✓ CONFIRMED |
| Learnable 1D pos enc ≈ fixed 2D sinusoidal | Ablation: -1.1% with sinusoidal | ✓ CONFIRMED |
| CLS token and GAP perform comparably | Ablation: -0.4% with GAP | ✓ CONFIRMED |

---

## Structural Limitations

1. **O(N²) Attention Complexity** — Memory and compute scale quadratically with sequence length. High-resolution images become prohibitive.
2. **No Spatial Inductive Bias** — ViT must learn locality and translation equivariance from data; needs 14M+ images to match CNNs.
3. **Fixed Training Resolution** — Positional embeddings are tied to the training grid; changing resolution requires interpolation.
4. **Uniform Patch Size** — No multi-scale hierarchy; limits performance on detection/segmentation vs. CNNs.

---

## Hardware Requirements

| Experiment | GPU VRAM | Time (approx.) |
|---|---|---|
| ViT-Tiny from scratch | 4 GB | ~4-6 hours |
| ViT-Tiny + augmentation | 4 GB | ~5-7 hours |
| ViT-Base fine-tuning | 8 GB | ~1 hour |
| ResNet-18 baseline | 2 GB | ~2 hours |
| All ablations | 4 GB | ~20-24 hours total |

CPU training is supported but much slower. Recommended: any NVIDIA GPU with 8+ GB VRAM.

---

## Citation

```bibtex
@inproceedings{dosovitskiy2021image,
  title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and
          Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and
          Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and
          Gelly, Sylvain and Uszkoreit, Jakob and Houlsby, Neil},
  booktitle={International Conference on Learning Representations},
  year={2021},
  url={https://openreview.net/forum?id=YicbFdNTTy}
}
```



*Reproducibility study by [@marviiiin](https://github.com/marviiiin) · April 2026*


