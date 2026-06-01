"""
train.py
========
Full training loop for VisionAST U-Net.

Usage:
    python src/train.py \
        --dryad_images   data/dryad/images \
        --dryad_masks    data/dryad/masks \
        --zip_images     data/zip_dataset/images \
        --zip_masks      data/zip_dataset/masks \
        --epochs         50 \
        --batch_size     4 \
        --img_size       512 \
        --model_size     small \
        --output_dir     models/
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

# local imports
import sys
sys.path.insert(0, str(Path(__file__).parent))
from dataset import make_splits
from model   import build_model
from losses  import BCEDiceLoss


# ── metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(preds: torch.Tensor, targets: torch.Tensor):
    """
    preds   : (B, H, W) predicted class ids (long)
    targets : (B, H, W) ground-truth class ids (long)
    Returns dict with pixel_acc, iou, dice
    """
    preds   = preds.cpu().numpy().flatten()
    targets = targets.cpu().numpy().flatten()

    tp = ((preds == 1) & (targets == 1)).sum()
    fp = ((preds == 1) & (targets == 0)).sum()
    fn = ((preds == 0) & (targets == 1)).sum()
    tn = ((preds == 0) & (targets == 0)).sum()

    pixel_acc = (tp + tn) / (tp + fp + fn + tn + 1e-8)
    iou       = tp / (tp + fp + fn + 1e-8)
    dice      = 2 * tp / (2 * tp + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)

    return {
        "pixel_acc": float(pixel_acc),
        "iou":       float(iou),
        "dice":      float(dice),
        "precision": float(precision),
        "recall":    float(recall),
    }


# ── training loop ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = 0.0

    for imgs, masks in loader:
        imgs  = imgs.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()

        if scaler:                          # AMP (GPU only)
            with torch.cuda.amp.autocast():
                logits = model(imgs)
                loss   = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss   = criterion(logits, masks)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_metrics = {"pixel_acc": [], "iou": [], "dice": [],
                   "precision": [], "recall": []}

    for imgs, masks in loader:
        imgs  = imgs.to(device)
        masks = masks.to(device)
        logits = model(imgs)
        loss   = criterion(logits, masks)
        total_loss += loss.item()

        preds = logits.argmax(dim=1)
        m     = compute_metrics(preds, masks)
        for k, v in m.items():
            all_metrics[k].append(v)

    avg_loss    = total_loss / len(loader)
    avg_metrics = {k: float(np.mean(v)) for k, v in all_metrics.items()}
    return avg_loss, avg_metrics


# ── main ──────────────────────────────────────────────────────────────────────

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── data ─────────────────────────────────────────────────────────────
    images_dirs = []
    masks_dirs  = []

    if args.dryad_images and args.dryad_masks:
        images_dirs.append(args.dryad_images)
        masks_dirs.append(args.dryad_masks)
    if args.zip_images and args.zip_masks:
        images_dirs.append(args.zip_images)
        masks_dirs.append(args.zip_masks)

    if not images_dirs:
        print("❌  No dataset paths provided. Use --dryad_images/--zip_images")
        return

    train_ds, val_ds, test_ds = make_splits(
        images_dirs, masks_dirs,
        img_size=(args.img_size, args.img_size),
        val_frac=0.15, test_frac=0.10
    )

    # Disable num_workers and pin_memory on CPU to avoid hangs
    use_workers  = 2 if device == "cuda" else 0
    use_pin_mem  = True if device == "cuda" else False
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=use_workers, pin_memory=use_pin_mem)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=use_workers, pin_memory=use_pin_mem)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, num_workers=use_workers)

    # ── model ─────────────────────────────────────────────────────────────
    model     = build_model(size=args.model_size, device=device)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None

    # ── output dir ────────────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = out_dir / "best_model.pth"
    last_ckpt = out_dir / "last_model.pth"

    # ── training ──────────────────────────────────────────────────────────
    best_val_dice = 0.0
    patience      = args.patience
    patience_ctr  = 0
    history       = []

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Val Loss':>8}  "
          f"{'Val IoU':>7}  {'Val Dice':>8}  {'LR':>8}")
    print("─" * 65)

    for epoch in range(1, args.epochs + 1):
        t0          = time.time()
        train_loss  = train_epoch(model, train_loader, optimizer,
                                  criterion, device, scaler)
        val_loss, vm = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        print(f"{epoch:>6}  {train_loss:>10.4f}  {val_loss:>8.4f}  "
              f"{vm['iou']:>7.4f}  {vm['dice']:>8.4f}  {lr:>8.2e}  "
              f"({time.time()-t0:.0f}s)")

        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss, **vm})

        # checkpoint
        torch.save({"epoch": epoch, "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_dice": vm["dice"]}, last_ckpt)

        if vm["dice"] > best_val_dice:
            best_val_dice = vm["dice"]
            torch.save(model.state_dict(), best_ckpt)
            print(f"         ✅  New best val Dice: {best_val_dice:.4f}")
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"\n⏹  Early stopping at epoch {epoch} "
                      f"(no improvement for {patience} epochs).")
                break

    # ── test evaluation ───────────────────────────────────────────────────
    print("\n── Test Evaluation ──")
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    _, test_metrics = eval_epoch(model, test_loader, criterion, device)
    for k, v in test_metrics.items():
        print(f"  {k:>12}: {v:.4f}")

    # ── save training history ─────────────────────────────────────────────
    import csv
    history_path = out_dir / "training_history.csv"
    with open(history_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    print(f"\nHistory saved: {history_path}")
    print(f"Best model  : {best_ckpt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VisionAST U-Net")
    # data
    parser.add_argument("--dryad_images", default=None)
    parser.add_argument("--dryad_masks",  default=None)
    parser.add_argument("--zip_images",   default=None)
    parser.add_argument("--zip_masks",    default=None)
    # training
    parser.add_argument("--epochs",      type=int,   default=50)
    parser.add_argument("--batch_size",  type=int,   default=4)
    parser.add_argument("--img_size",    type=int,   default=512)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--patience",    type=int,   default=10)
    parser.add_argument("--model_size",  choices=["full","small"],
                        default="small",
                        help="small=32 base filters (CPU friendly)")
    parser.add_argument("--output_dir",  default="models/")
    args = parser.parse_args()
    main(args)
