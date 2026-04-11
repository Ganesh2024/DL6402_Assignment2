"""Ablation training script for W&B report sections 2.1 and 2.2.

Section 2.1 — BatchNorm effect
    Trains two models (with BN, without BN) for 20 epochs each.
    Logs activation distributions of the 3rd conv layer.
    Compares convergence speed and maximum stable learning rate.

Section 2.2 — Dropout effect
    Trains three models (p=0, p=0.2, p=0.5) for 20 epochs each.
    Overlays train vs val loss curves for all three.

Usage
-----
python train_ablation.py \
    --data_root /kaggle/working/pets \
    --ckpt_dir  /kaggle/working/checkpoints \
    --epochs    20 \
    --batch_size 64
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import f1_score
import wandb

from data.pets_dataset import OxfordIIITPetDataset, build_transforms
from models.vgg11_ablation import VGG11ClassifierAblation


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def collate_fn(batch):
    images    = torch.stack([b["image"]    for b in batch])
    class_ids = torch.stack([b["class_id"] for b in batch])
    return {"image": images, "class_id": class_ids,
            "bbox": None, "mask": None}


def make_loaders(data_root, batch_size):
    train_tf = build_transforms(is_train=True)
    val_tf   = build_transforms(is_train=False)
    full_ds  = OxfordIIITPetDataset(
        data_root, split="trainval",
        transform=train_tf, task="classification"
    )
    n_val   = max(1, int(0.15 * len(full_ds)))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    val_ds.dataset.transform = val_tf
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=2, collate_fn=collate_fn, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, collate_fn=collate_fn, pin_memory=True
    )
    return train_loader, val_loader


def get_third_conv_activations(model, sample_img):
    """
    Extract activations from the 3rd convolutional layer (stage2[0][0]).
    Used for Section 2.1 activation distribution plots.
    """
    activations = {}

    def hook_fn(module, inp, out):
        activations["third_conv"] = out.detach().cpu()

    # 3rd conv layer is stage2's conv block (conv1→stage1, conv2→stage2)
    hook = model.backbone.stage2[0][0].register_forward_hook(hook_fn)

    model.eval()
    with torch.no_grad():
        model(sample_img.to(DEVICE))
    hook.remove()

    return activations["third_conv"]


def run_one_epoch(model, loader, criterion, optimiser, is_train):
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in loader:
            imgs   = batch["image"].to(DEVICE)
            labels = batch["class_id"].to(DEVICE)

            if is_train:
                optimiser.zero_grad()

            logits = model(imgs)
            loss   = criterion(logits, labels)

            if is_train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimiser.step()

            total_loss += loss.item() * imgs.size(0)
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += imgs.size(0)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / total
    acc      = correct / total
    f1       = f1_score(all_labels, all_preds,
                        average="macro", zero_division=0)
    return avg_loss, acc, f1


# ---------------------------------------------------------------------------
# Single ablation run
# ---------------------------------------------------------------------------

def run_ablation(
    run_name, data_root, batch_size, epochs,
    lr, dropout_p, use_bn, ckpt_dir
):
    """Train one ablation model and log everything to W&B."""
    print(f"\n{'='*55}")
    print(f"  Ablation run: {run_name}")
    print(f"  use_bn={use_bn}  dropout_p={dropout_p}  lr={lr}")
    print(f"{'='*55}")

    train_loader, val_loader = make_loaders(data_root, batch_size)

    model = VGG11ClassifierAblation(
        num_classes=37,
        dropout_p=dropout_p,
        use_bn=use_bn,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=5e-3
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="max", factor=0.5,
        patience=4, min_lr=1e-6
    )

    wandb.init(
        project="DA6401-A2",
        name=run_name,
        config={
            "use_bn": use_bn,
            "dropout_p": dropout_p,
            "lr": lr,
            "epochs": epochs,
            "batch_size": batch_size,
        },
        reinit="finish_previous"
    )

    # Get a fixed sample image for activation logging
    sample_batch = next(iter(val_loader))
    sample_img   = sample_batch["image"][:1]

    best_f1 = 0.0

    for epoch in range(1, epochs + 1):
        # Warmup LR for first 3 epochs
        if epoch <= 3:
            scale = epoch / 3
            for pg in optimiser.param_groups:
                pg["lr"] = lr * scale

        tr_loss, tr_acc, tr_f1 = run_one_epoch(
            model, train_loader, criterion, optimiser, is_train=True
        )
        vl_loss, vl_acc, vl_f1 = run_one_epoch(
            model, val_loader, criterion, optimiser, is_train=False
        )

        if epoch > 3:
            scheduler.step(vl_f1)

        cur_lr = optimiser.param_groups[0]["lr"]

        print(
            f"  Ep {epoch:3d} | "
            f"tr {tr_loss:.3f}/{tr_acc:.3f}/f1={tr_f1:.3f} | "
            f"val {vl_loss:.3f}/{vl_acc:.3f}/f1={vl_f1:.3f} | "
            f"lr {cur_lr:.2e}"
        )

        # Log activation distribution of 3rd conv layer every 5 epochs
        act = get_third_conv_activations(model, sample_img)
        act_flat = act.flatten().numpy()

        log_dict = {
            "train/loss":     tr_loss,
            "train/acc":      tr_acc,
            "train/f1":       tr_f1,
            "val/loss":       vl_loss,
            "val/acc":        vl_acc,
            "val/f1":         vl_f1,
            "lr":             cur_lr,
            "epoch":          epoch,
            # Activation stats for Section 2.1
            "activations/mean":  float(act_flat.mean()),
            "activations/std":   float(act_flat.std()),
            "activations/hist":  wandb.Histogram(act_flat),
        }
        wandb.log(log_dict)

        if vl_f1 > best_f1:
            best_f1 = vl_f1

    print(f"  Best val F1: {best_f1:.4f}")
    wandb.finish()
    return best_f1


# ---------------------------------------------------------------------------
# Main — run all ablation experiments
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",  type=str,
                   default="/kaggle/working/pets")
    p.add_argument("--ckpt_dir",   type=str,
                   default="/kaggle/working/checkpoints")
    p.add_argument("--epochs",     type=int,   default=20)
    p.add_argument("--batch_size", type=int,   default=64)
    p.add_argument("--lr",         type=float, default=3e-4)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.ckpt_dir, exist_ok=True)

    results = {}

    # ── Section 2.1: BatchNorm ablation ──────────────────────────
    print("\n" + "="*55)
    print("  SECTION 2.1 — BatchNorm Ablation")
    print("="*55)

    # With BN (standard model)
    results["with_bn"] = run_ablation(
        run_name   = "ablation-with-bn",
        data_root  = args.data_root,
        batch_size = args.batch_size,
        epochs     = args.epochs,
        lr         = args.lr,
        dropout_p  = 0.5,
        use_bn     = True,
        ckpt_dir   = args.ckpt_dir,
    )

    # Without BN
    results["without_bn"] = run_ablation(
        run_name   = "ablation-no-bn",
        data_root  = args.data_root,
        batch_size = args.batch_size,
        epochs     = args.epochs,
        lr         = args.lr,
        dropout_p  = 0.5,
        use_bn     = False,
        ckpt_dir   = args.ckpt_dir,
    )

    # ── Section 2.2: Dropout ablation ────────────────────────────
    print("\n" + "="*55)
    print("  SECTION 2.2 — Dropout Ablation")
    print("="*55)

    for dp in [0.0, 0.2, 0.5]:
        name = f"ablation-dropout-p{int(dp*10)}"
        results[f"dropout_{dp}"] = run_ablation(
            run_name   = name,
            data_root  = args.data_root,
            batch_size = args.batch_size,
            epochs     = args.epochs,
            lr         = args.lr,
            dropout_p  = dp,
            use_bn     = True,
            ckpt_dir   = args.ckpt_dir,
        )

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "="*55)
    print("  ABLATION RESULTS SUMMARY")
    print("="*55)
    for name, f1 in results.items():
        print(f"  {name:<25} best_val_f1 = {f1:.4f}")
