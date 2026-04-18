"""
CIFAR-10 DataLoaders
====================
Provides train / val / test DataLoaders for CIFAR-10.

Dataset statistics:
  - 50,000 training samples  (10 classes × 5,000)
  - 10,000 test samples      (10 classes × 1,000)
  - Image size: 32×32 RGB
  - Classes: airplane, automobile, bird, cat, deer,
             dog, frog, horse, ship, truck

We carve out 5,000 samples from the training set as a validation split
to monitor overfitting during training, keeping the test set for final
evaluation only.
"""

import os
import torch
from torch.utils.data import DataLoader, random_split
import torchvision.datasets as datasets
from .transforms import (
    get_train_transform, get_val_transform,
    get_finetune_train_transform, get_finetune_val_transform,
)

class TransformSubset(torch.utils.data.Dataset):
    """Dataset wrapper that applies a transform to a subset of another dataset.
    Defined at module level so it is picklable by Windows multiprocessing spawn."""
    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = list(indices)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, label = self.dataset[self.indices[idx]]
        if self.transform:
            img = self.transform(img)
        return img, label


CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def get_cifar10_loaders(
    data_dir: str = "./data/cifar10",
    img_size: int = 32,
    batch_size: int = 256,
    num_workers: int = 4,
    val_size: int = 5000,
    seed: int = 42,
    use_rand_augment: bool = False,
    use_random_erasing: bool = True,
    finetune_mode: bool = False,
):
    """
    Build and return (train_loader, val_loader, test_loader).

    Args:
        data_dir         : Root directory for downloading CIFAR-10.
        img_size         : Target resolution (32 for from-scratch, 224 for fine-tuning).
        batch_size       : Samples per batch.
        num_workers      : DataLoader worker processes.
        val_size         : Number of training samples held out as validation.
        seed             : Random seed for train/val split reproducibility.
        use_rand_augment : Enable RandAugment in training transforms.
        use_random_erasing: Enable Random Erasing in training transforms.
        finetune_mode    : Use 224×224 transforms and ImageNet normalization stats.
                           Set True when fine-tuning a pretrained ViT.

    Returns:
        train_loader, val_loader, test_loader
    """
    os.makedirs(data_dir, exist_ok=True)

    if finetune_mode:
        train_tf = get_finetune_train_transform(img_size)
        val_tf   = get_finetune_val_transform(img_size)
    else:
        train_tf = get_train_transform(
            img_size, use_rand_augment, use_random_erasing,
            normalize_imagenet=False,
        )
        val_tf = get_val_transform(img_size, normalize_imagenet=False)

    # Load full training set (50k) without transforms first for splitting
    full_train = datasets.CIFAR10(data_dir, train=True, download=True, transform=None)
    test_set   = datasets.CIFAR10(data_dir, train=False, download=True, transform=val_tf)

    # Reproducible train/val split
    generator = torch.Generator().manual_seed(seed)
    train_indices, val_indices = random_split(
        range(len(full_train)), [len(full_train) - val_size, val_size],
        generator=generator,
    )

    # Wrap with transforms using module-level class (required for Windows multiprocessing pickle)
    from torch.utils.data import Subset

    train_set = TransformSubset(full_train, train_indices, train_tf)
    val_set   = TransformSubset(full_train, val_indices,   val_tf)

    # num_workers=0 on Windows avoids spawn/pickle issues with DataLoader workers
    nw = 0 if os.name == "nt" else num_workers
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=nw, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=nw, pin_memory=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=nw, pin_memory=True,
    )

    print(f"[Data] CIFAR-10 | Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")
    return train_loader, val_loader, test_loader
