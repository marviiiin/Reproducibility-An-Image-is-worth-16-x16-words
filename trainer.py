"""
Main Training Loop
==================
Encapsulates training and validation logic with:
  - Mixed precision (torch.cuda.amp) for memory efficiency
  - Gradient clipping to prevent exploding gradients
  - TensorBoard logging for training curves
  - CSV logging for portability
  - Checkpoint saving (best val acc + last epoch)
  - Mixup/CutMix support via soft target loss
"""

import os
import csv
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Optional, Callable

from .losses import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from .scheduler import WarmupCosineScheduler


class Trainer:
    """
    Args:
        model           : VisionTransformer or any nn.Module.
        train_loader    : Training DataLoader.
        val_loader      : Validation DataLoader.
        optimizer       : AdamW or similar.
        scheduler       : LR scheduler (WarmupCosineScheduler).
        num_epochs      : Total training epochs.
        device          : 'cuda' or 'cpu'.
        label_smoothing : Smoothing ε for LabelSmoothingCrossEntropy.
        grad_clip       : Max gradient norm (1.0 default).
        log_dir         : Directory for TensorBoard and CSV logs.
        checkpoint_dir  : Directory for saving .pth checkpoints.
        mixup_fn        : Optional MixupCutmix callable. If provided, uses SoftTargetCE.
        experiment_name : Name prefix for checkpoints and logs.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: WarmupCosineScheduler,
        num_epochs: int = 300,
        device: str = "cuda",
        label_smoothing: float = 0.1,
        grad_clip: float = 1.0,
        log_dir: str = "results/logs",
        checkpoint_dir: str = "checkpoints",
        mixup_fn: Optional[Callable] = None,
        experiment_name: str = "vit",
    ):
        self.model            = model.to(device)
        self.train_loader     = train_loader
        self.val_loader       = val_loader
        self.optimizer        = optimizer
        self.scheduler        = scheduler
        self.num_epochs       = num_epochs
        self.device           = device
        self.grad_clip        = grad_clip
        self.mixup_fn         = mixup_fn
        self.experiment_name  = experiment_name

        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.checkpoint_dir = checkpoint_dir

        # Loss functions
        self.hard_criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
        self.soft_criterion = SoftTargetCrossEntropy()

        # Mixed precision scaler
        self.scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=os.path.join(log_dir, experiment_name))

        # CSV logger
        self.csv_path = os.path.join(log_dir, f"{experiment_name}_metrics.csv")
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss", "val_acc_top1",
                             "val_acc_top5", "lr", "epoch_time_s"])

        self.best_val_acc = 0.0
        self.history = {"train_loss": [], "val_loss": [], "val_acc_top1": [], "val_acc_top5": []}

    # ─── Training Epoch ──────────────────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Train]",
                    leave=False, ncols=100)

        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            # Apply Mixup/CutMix if configured
            if self.mixup_fn is not None:
                images, soft_labels = self.mixup_fn(images, labels)
            else:
                soft_labels = None

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(self.device == "cuda")):
                logits, _ = self.model(images)
                if soft_labels is not None:
                    loss = self.soft_criterion(logits, soft_labels)
                else:
                    loss = self.hard_criterion(logits, labels)

            self.scaler.scale(loss).backward()

            # Gradient clipping (unscale first so clip operates on true gradients)
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / num_batches

    # ─── Validation Epoch ────────────────────────────────────────────────────

    @torch.no_grad()
    def _val_epoch(self) -> tuple:
        self.model.eval()
        total_loss = 0.0
        correct_top1 = 0
        correct_top5 = 0
        total = 0

        for images, labels in tqdm(self.val_loader, desc="  [Val]", leave=False, ncols=100):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=(self.device == "cuda")):
                logits, _ = self.model(images)
                loss = self.hard_criterion(logits, labels)

            total_loss += loss.item()

            # Top-1 accuracy
            _, pred_top1 = logits.topk(1, dim=1)
            correct_top1 += pred_top1.eq(labels.view(-1, 1)).sum().item()

            # Top-5 accuracy
            k = min(5, logits.size(1))
            _, pred_topk = logits.topk(k, dim=1)
            correct_top5 += pred_topk.eq(labels.view(-1, 1).expand_as(pred_topk)).sum().item()

            total += labels.size(0)

        avg_loss  = total_loss / len(self.val_loader)
        acc_top1  = 100.0 * correct_top1 / total
        acc_top5  = 100.0 * correct_top5 / total
        return avg_loss, acc_top1, acc_top5

    # ─── Save Checkpoint ─────────────────────────────────────────────────────

    def _save_checkpoint(self, epoch: int, val_acc: float, is_best: bool):
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_acc": val_acc,
            "scaler_state_dict": self.scaler.state_dict(),
        }
        last_path = os.path.join(self.checkpoint_dir, f"{self.experiment_name}_last.pth")
        torch.save(state, last_path)

        if is_best:
            best_path = os.path.join(self.checkpoint_dir, f"{self.experiment_name}_best.pth")
            torch.save(state, best_path)
            print(f"  [BEST] New best model saved: {val_acc:.2f}%")

    # ─── Main Training Loop ───────────────────────────────────────────────────

    def train(self):
        print(f"\n{'='*60}")
        print(f"  Experiment : {self.experiment_name}")
        print(f"  Device     : {self.device}")
        print(f"  Epochs     : {self.num_epochs}")
        print(f"  Params     : {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        print(f"{'='*60}\n")

        for epoch in range(self.num_epochs):
            t0 = time.time()

            # Train
            train_loss = self._train_epoch(epoch)

            # Validate
            val_loss, val_acc_top1, val_acc_top5 = self._val_epoch()

            # LR step
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            epoch_time = time.time() - t0

            # Log to TensorBoard
            self.writer.add_scalar("Loss/Train", train_loss, epoch)
            self.writer.add_scalar("Loss/Val",   val_loss,   epoch)
            self.writer.add_scalar("Acc/Top1",   val_acc_top1, epoch)
            self.writer.add_scalar("Acc/Top5",   val_acc_top5, epoch)
            self.writer.add_scalar("LR",         current_lr,   epoch)

            # Log to CSV
            with open(self.csv_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    epoch + 1, f"{train_loss:.6f}", f"{val_loss:.6f}",
                    f"{val_acc_top1:.4f}", f"{val_acc_top5:.4f}",
                    f"{current_lr:.8f}", f"{epoch_time:.1f}"
                ])

            # History for plotting
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc_top1"].append(val_acc_top1)
            self.history["val_acc_top5"].append(val_acc_top5)

            # Save checkpoint
            is_best = val_acc_top1 > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc_top1
            self._save_checkpoint(epoch, val_acc_top1, is_best)

            print(
                f"Epoch {epoch+1:3d}/{self.num_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Top-1: {val_acc_top1:.2f}% | "
                f"Top-5: {val_acc_top5:.2f}% | "
                f"LR: {current_lr:.6f} | "
                f"Time: {epoch_time:.0f}s"
            )

        self.writer.close()
        print(f"\nTraining complete. Best Val Top-1: {self.best_val_acc:.2f}%")
        return self.history
