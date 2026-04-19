
# Reproducing "An Image is Worth 16×16 Words"
### Vision Transformer (ViT) — PyTorch Implementation on CIFAR-10

> **Paper:** Dosovitskiy et al. (2021), *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*, ICLR 2021. [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)

A reproduction of the Vision Transformer (ViT), evaluated on CIFAR-10 with 5 systematic experiments.

---
The model was evaluated on CIFAR - 10 dataset

![image](https://github.com/marviiiin/Reproducibility-An-Image-is-worth-16-x16-words/blob/c229a95518337842aad172c45b2d0c67beeb2562/cifar-10.png)


## Key Results

| Model | Dataset | Test Top-1 | 
| **ViT-Base/16 (pretrained)** | CIFAR-10 | **98.68%** | 
| ViT-Base/16 (original paper) | CIFAR-10 | 98.95% |


---

## Architecture Overview
![image](https://github.com/marviiiin/Reproducibility-An-Image-is-worth-16-x16-words/blob/443181da50d96853210f51759f8fc0d61639866b/vit_figure.png)
```
Image (B, 3, 224, 224)
       ↓
Patch Embedding (Conv2d P=16, stride=16)
       ↓
[CLS] token prepended → (B, 197, 768)
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
LayerNorm → CLS token → Linear(768, 10)
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

# 3. Experiment 1: Fine-tune pretrained ViT-Base (recommended first run)
python scripts/train.py --config configs/vit_base_finetune_cifar10.yaml



---


---

### CIFAR-10 Adaptation

  So the original paper takes 32×32 CIFAR-10 images and upsamples them all the way to 384×384 — a 12× increase in resolution. At 384×384 with patch size 16, that gives 576 patches (24×24 grid)
 We upsampled to 224×224 instead, giving only 196 patches (14×14 grid) — significantly fewer spatial tokens for the model to work with.                                                                                                                                       
A likely contributor to our result (98.68%) being 0.27 pp below the paper (98.95%) is the reduced patch count — fewer spatial tokens mean less granular information per image..     
---

---

## Structural Limitations

1. **No Spatial Inductive Bias** — ViT must learn locality and translation equivariance from data; needs 14M+ images to match CNNs.
2. **Fixed Training Resolution** — Positional embeddings are tied to the training grid; changing resolution requires interpolation.
3. **Uniform Patch Size** — No multi-scale hierarchy; limits performance on detection/segmentation vs. CNNs.


---

## Hardware Requirements

| Experiment | GPU VRAM | Time (approx.) |
|---|---|---|
| ViT-Base fine-tuning | 8 GB | ~1 hour |



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


