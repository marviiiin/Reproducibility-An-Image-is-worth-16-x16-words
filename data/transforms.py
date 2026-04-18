"""
Data Augmentation Pipelines
============================
Mirrors the training augmentation strategy described in:
  - ViT paper (Appendix B.1): random crop + horizontal flip
  - DeiT (Touvron et al., 2021): RandAugment + Mixup + CutMix + Random Erasing

CIFAR-10 normalization stats (computed from the full training set):
  mean = [0.4914, 0.4822, 0.4465]
  std  = [0.2470, 0.2435, 0.2616]
"""

import torchvision.transforms as T
from typing import Tuple

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

# ImageNet stats (use when fine-tuning a pretrained ViT on CIFAR-10)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def get_train_transform(
    img_size: int = 32,
    use_rand_augment: bool = False,
    use_random_erasing: bool = True,
    normalize_imagenet: bool = False,
) -> T.Compose:
    """
    Training transform pipeline.

    Args:
        img_size            : Target image size after crop.
        use_rand_augment    : Apply RandAugment (N=2, M=9). Improves from-scratch training.
        use_random_erasing  : Apply Random Erasing (p=0.25). Regularization.
        normalize_imagenet  : Use ImageNet stats (for fine-tuning pretrained ViT).
    """
    mean = IMAGENET_MEAN if normalize_imagenet else CIFAR10_MEAN
    std  = IMAGENET_STD  if normalize_imagenet else CIFAR10_STD

    transforms = [
        T.RandomCrop(img_size, padding=4),
        T.RandomHorizontalFlip(p=0.5),
    ]

    if use_rand_augment:
        transforms.append(T.RandAugment(num_ops=2, magnitude=9))

    transforms += [
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ]

    if use_random_erasing:
        transforms.append(T.RandomErasing(p=0.25))

    return T.Compose(transforms)


def get_val_transform(
    img_size: int = 32,
    normalize_imagenet: bool = False,
) -> T.Compose:
    """
    Validation / test transform pipeline (deterministic).
    """
    mean = IMAGENET_MEAN if normalize_imagenet else CIFAR10_MEAN
    std  = IMAGENET_STD  if normalize_imagenet else CIFAR10_STD

    return T.Compose([
        T.Resize(img_size),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


def get_finetune_train_transform(img_size: int = 224) -> T.Compose:
    """
    Stronger augmentation for fine-tuning pretrained ViT on CIFAR-10 at 224x224.
    Matches the recipe from Steiner et al. 2021 'How to train your ViT?'
    """
    return T.Compose([
        T.Resize(img_size + 32),
        T.RandomCrop(img_size),
        T.RandomHorizontalFlip(p=0.5),
        T.RandAugment(num_ops=2, magnitude=9),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        T.RandomErasing(p=0.25),
    ])


def get_finetune_val_transform(img_size: int = 224) -> T.Compose:
    return T.Compose([
        T.Resize(int(img_size * 1.15)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
