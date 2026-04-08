"""Training entry-point for all three tasks.

Usage
-----
python train.py --task classification --data_root /content/pets \
    --epochs 30 --lr 3e-4 --batch_size 64 --dropout_p 0.4

python train.py --task localization --data_root /content/pets \
    --epochs 40 --lr 1e-4 --batch_size 32 --dropout_p 0.4

python train.py --task segmentation --unet_mode frozen \
    --data_root /content/pets --epochs 30 --lr 1e-4 --batch_size 16

python train.py --task segmentation --unet_mode partial \
    --data_root /content/pets --epochs 30 --lr 1e-4 --batch_size 16

python train.py --task segmentation --unet_mode full \
    --data_root /content/pets --epochs 30 --lr 1e-4 --batch_size 16
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
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet
from losses.iou_loss import IoULoss


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def get_ckpt_dir(args) -> str:
    os.makedirs(args.ckpt_dir, exist_ok=True)
    return args.ckpt_dir


def save_ckpt(model, epoch, metric, fpath):
    torch.save(
        {"state_dict": model.state_dict(),
         "epoch": epoch, "best_metric": metric},
        fpath,
    )
    print(f"  → Saved {fpath}  (epoch={epoch}, metric={metric:.4f})")


def collate_fn(batch):
    images    = torch.stack([b["image"]    for b in batch])
    class_ids = torch.stack([b["class_id"] for b in batch])
    bboxes = (
        torch.stack([b["bbox"] for b in batch])
        if all(b["bbox"] is not None for b in batch) else None
    )
    masks = (
        torch.stack([b["mask"] for b in batch])
        if all(b["mask"] is not None for b in batch) else None
    )
    return {"image": images, "class_id": class_ids,
            "bbox": bboxes, "mask": masks}


def make_loaders(args, task="all"):
    train_tf = build_transforms(is_train=True)
    val_tf   = build_transforms(is_train=False)
    full_ds  = OxfordIIITPetDataset(
        args.data_root, split="trainval",
        transform=train_tf, task=task
    )
    n_val   = max(1, int(0.15 * len(full_ds)))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    val_ds.dataset.transform = val_tf
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=2, collate_fn=collate_fn, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=2, collate_fn=collate_fn, pin_memory=True
    )
    return train_loader, val_loader


def evaluate_classifier(model, val_loader, criterion):
    """Run one validation pass, return (loss, accuracy, macro_f1)."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            imgs   = batch["image"].to(DEVICE)
            labels = batch["class_id"].to(DEVICE)
            logits = model(imgs)
            total_loss += criterion(logits, labels).item() * imgs.size(0)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    n        = len(all_labels)
    val_loss = total_loss / n
    val_acc  = np.mean(np.array(all_preds) == np.array(all_labels))
    val_f1   = f1_score(all_labels, all_preds,
                        average="macro", zero_division=0)
    return val_loss, val_acc, val_f1


def one_train_epoch_clf(model, loader, criterion, optimiser):
    model.train()
    run_loss, correct, total = 0.0, 0, 0
    for batch in loader:
        imgs   = batch["image"].to(DEVICE)
        labels = batch["class_id"].to(DEVICE)
        optimiser.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimiser.step()
        run_loss += loss.item() * imgs.size(0)
        correct  += (logits.argmax(1) == labels).sum().item()
        total    += imgs.size(0)
    return run_loss / total, correct / total


# ---------------------------------------------------------------------------
# Dice loss for segmentation
# ---------------------------------------------------------------------------
class DiceLoss(nn.Module):
    def __init__(self, num_classes=3, smooth=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.smooth      = smooth

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        total = 0.0
        for c in range(self.num_classes):
            p   = probs[:, c]
            g   = (targets == c).float()
            num = 2.0 * (p * g).sum() + self.smooth
            den = p.sum() + g.sum() + self.smooth
            total += 1.0 - num / den
        return total / self.num_classes


# ---------------------------------------------------------------------------
# Task 1 — Classification with two-phase training
# ---------------------------------------------------------------------------
def train_classifier(args):
    print("=" * 55)
    print("  TASK 1 — Classification  (pretrained backbone)")
    print("=" * 55)

    train_loader, val_loader = make_loaders(args, task="classification")
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # pretrained=True loads ImageNet VGG11-BN weights into the encoder
    model = VGG11Classifier(
        num_classes=37,
        dropout_p=args.dropout_p,
        pretrained=True,
    ).to(DEVICE)

    if not args.no_wandb:
        wandb.init(project="DA6401-A2", name="classifier",
                   config=vars(args), reinit="finish_previous")

    best_f1   = 0.0
    ckpt_path = os.path.join(get_ckpt_dir(args), "classifier.pth")

    # ── Phase 1: freeze backbone, train FC head only ─────────────
    phase1_epochs = min(10, args.epochs // 3)
    print(f"\n── Phase 1: backbone frozen ({phase1_epochs} warm-up epochs) ──")
    for p in model.backbone.parameters():
        p.requires_grad = False

    opt1 = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-3
    )
    sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt1, T_max=phase1_epochs, eta_min=1e-6
    )

    for epoch in range(1, phase1_epochs + 1):
        tr_loss, tr_acc = one_train_epoch_clf(model, train_loader,
                                               criterion, opt1)
        sch1.step()
        vl, va, vf = evaluate_classifier(model, val_loader, criterion)
        print(f"  [P1] Ep {epoch:2d} | "
              f"tr {tr_loss:.3f}/{tr_acc:.3f} | "
              f"val {vl:.3f}/{va:.3f}/f1={vf:.3f}")
        if not args.no_wandb:
            wandb.log({"clf/train_loss": tr_loss, "clf/train_acc": tr_acc,
                       "clf/val_loss": vl, "clf/val_acc": va,
                       "clf/val_f1": vf, "epoch": epoch})
        if vf > best_f1:
            best_f1 = vf
            save_ckpt(model, epoch, best_f1, ckpt_path)

    # ── Phase 2: unfreeze all, fine-tune end-to-end ──────────────
    phase2_epochs = args.epochs - phase1_epochs
    print(f"\n── Phase 2: full fine-tune ({phase2_epochs} epochs) ──")
    for p in model.backbone.parameters():
        p.requires_grad = True

    # Backbone gets 10x lower LR to preserve pretrained features
    opt2 = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": args.lr * 0.1},
        {"params": model.fc_head.parameters(),  "lr": args.lr},
    ], weight_decay=5e-3)
    sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt2, T_max=phase2_epochs, eta_min=1e-7
    )

    for epoch in range(phase1_epochs + 1, args.epochs + 1):
        tr_loss, tr_acc = one_train_epoch_clf(model, train_loader,
                                               criterion, opt2)
        sch2.step()
        vl, va, vf = evaluate_classifier(model, val_loader, criterion)
        print(f"  [P2] Ep {epoch:2d} | "
              f"tr {tr_loss:.3f}/{tr_acc:.3f} | "
              f"val {vl:.3f}/{va:.3f}/f1={vf:.3f}")
        if not args.no_wandb:
            wandb.log({"clf/train_loss": tr_loss, "clf/train_acc": tr_acc,
                       "clf/val_loss": vl, "clf/val_acc": va,
                       "clf/val_f1": vf, "epoch": epoch})
        if vf > best_f1:
            best_f1 = vf
            save_ckpt(model, epoch, best_f1, ckpt_path)

    if not args.no_wandb:
        wandb.finish()
    print(f"\nBest val F1: {best_f1:.4f}")


# ---------------------------------------------------------------------------
# Task 2 — Localisation
# ---------------------------------------------------------------------------
def train_localizer(args):
    print("=" * 55)
    print("  TASK 2 — Localisation")
    print("=" * 55)

    train_loader, val_loader = make_loaders(args, task="localization")
    model = VGG11Localizer(
        dropout_p=args.dropout_p,
        freeze_backbone=True
    ).to(DEVICE)

    clf_path = os.path.join(get_ckpt_dir(args), "classifier.pth")
    if os.path.exists(clf_path):
        payload = torch.load(clf_path, map_location=DEVICE)
        sd      = payload.get("state_dict", payload)
        bb_sd   = {k.replace("backbone.", ""): v
                   for k, v in sd.items() if k.startswith("backbone.")}
        model.backbone.load_state_dict(bb_sd, strict=True)
        print("  → Backbone loaded from classifier.pth")
    else:
        print("  ⚠ classifier.pth not found — training backbone from scratch")

    mse_fn   = nn.MSELoss()
    iou_fn   = IoULoss(reduction="mean")
    iou_none = IoULoss(reduction="none")

    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-3
    )
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=1e-6
    )

    if not args.no_wandb:
        wandb.init(project="DA6401-A2", name="localizer",
                   config=vars(args), reinit="finish_previous")

    best_iou  = 0.0
    ckpt_path = os.path.join(get_ckpt_dir(args), "localizer.pth")

    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss, n = 0.0, 0
        for batch in train_loader:
            if batch["bbox"] is None:
                continue
            imgs  = batch["image"].to(DEVICE)
            boxes = batch["bbox"].to(DEVICE)
            opt.zero_grad()
            pred  = model(imgs)
            loss  = mse_fn(pred, boxes) + iou_fn(pred, boxes)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            run_loss += loss.item() * imgs.size(0)
            n        += imgs.size(0)
        sch.step()
        tr_loss = run_loss / max(n, 1)

        model.eval()
        val_loss, iou_sum, n_val = 0.0, 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                if batch["bbox"] is None:
                    continue
                imgs  = batch["image"].to(DEVICE)
                boxes = batch["bbox"].to(DEVICE)
                pred  = model(imgs)
                val_loss += (mse_fn(pred, boxes) +
                             iou_fn(pred, boxes)).item() * imgs.size(0)
                iou_sum  += (1.0 - iou_none(pred, boxes)).sum().item()
                n_val    += imgs.size(0)

        val_loss /= max(n_val, 1)
        val_iou   = iou_sum  / max(n_val, 1)
        print(f"Epoch {epoch:3d} | tr_loss {tr_loss:.4f} | "
              f"val_loss {val_loss:.4f}  val_iou {val_iou:.4f}")

        if not args.no_wandb:
            wandb.log({"loc/train_loss": tr_loss,
                       "loc/val_loss":   val_loss,
                       "loc/val_iou":    val_iou,
                       "epoch":          epoch})

        if val_iou > best_iou:
            best_iou = val_iou
            save_ckpt(model, epoch, best_iou, ckpt_path)

    if not args.no_wandb:
        wandb.finish()
    print(f"\nBest val IoU: {best_iou:.4f}")


# ---------------------------------------------------------------------------
# Task 3 — Segmentation  (frozen / partial / full)
# ---------------------------------------------------------------------------
def train_segmentation(args):
    mode = args.unet_mode
    print("=" * 55)
    print(f"  TASK 3 — Segmentation  [{mode}]")
    print("=" * 55)

    train_loader, val_loader = make_loaders(args, task="segmentation")
    model = VGG11UNet(num_classes=3, dropout_p=args.dropout_p).to(DEVICE)

    clf_path = os.path.join(get_ckpt_dir(args), "classifier.pth")
    if os.path.exists(clf_path):
        payload = torch.load(clf_path, map_location=DEVICE)
        sd      = payload.get("state_dict", payload)
        bb_sd   = {k.replace("backbone.", ""): v
                   for k, v in sd.items() if k.startswith("backbone.")}
        model.encoder.load_state_dict(bb_sd, strict=True)
        print("  → Encoder loaded from classifier.pth")

    # apply freezing strategy
    if mode == "frozen":
        for p in model.encoder.parameters():
            p.requires_grad = False
        print("  → Encoder fully frozen")
    elif mode == "partial":
        for p in model.encoder.parameters():
            p.requires_grad = False
        for stage in [model.encoder.stage3,
                      model.encoder.stage4,
                      model.encoder.stage5]:
            for p in stage.parameters():
                p.requires_grad = True
        print("  → Encoder partially frozen (stage1+2 frozen, stage3-5 trainable)")
    else:
        print("  → Full fine-tuning — all weights trainable")

    ce_fn   = nn.CrossEntropyLoss()
    dice_fn = DiceLoss(num_classes=3)

    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-3
    )
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=1e-6
    )

    if not args.no_wandb:
        wandb.init(project="DA6401-A2", name=f"unet-{mode}",
                   config=vars(args), reinit="finish_previous")

    best_dice = 0.0
    fname     = "unet.pth" if mode == "full" else f"unet_{mode}.pth"
    ckpt_path = os.path.join(get_ckpt_dir(args), fname)

    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss, n = 0.0, 0
        for batch in train_loader:
            if batch["mask"] is None:
                continue
            imgs  = batch["image"].to(DEVICE)
            masks = batch["mask"].to(DEVICE)
            opt.zero_grad()
            logits = model(imgs)
            loss   = 0.5 * ce_fn(logits, masks) + 0.5 * dice_fn(logits, masks)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            run_loss += loss.item() * imgs.size(0)
            n        += imgs.size(0)
        sch.step()
        tr_loss = run_loss / max(n, 1)

        model.eval()
        val_loss, dice_sum, pix_r, pix_t, n_val = 0.0, 0.0, 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                if batch["mask"] is None:
                    continue
                imgs   = batch["image"].to(DEVICE)
                masks  = batch["mask"].to(DEVICE)
                logits = model(imgs)
                val_loss += (0.5 * ce_fn(logits, masks) +
                             0.5 * dice_fn(logits, masks)).item() * imgs.size(0)
                preds   = logits.argmax(dim=1)
                pix_r  += (preds == masks).sum().item()
                pix_t  += masks.numel()
                probs   = torch.softmax(logits, dim=1)[:, 0]
                gt      = (masks == 0).float()
                num     = 2.0 * (probs * gt).sum() + 1.0
                den     = probs.sum() + gt.sum() + 1.0
                dice_sum += (num / den).item() * imgs.size(0)
                n_val    += imgs.size(0)

        val_loss /= max(n_val, 1)
        val_dice  = dice_sum / max(n_val, 1)
        val_pix   = pix_r   / max(pix_t,  1)
        print(f"Epoch {epoch:3d} | tr {tr_loss:.4f} | "
              f"val {val_loss:.4f}  dice {val_dice:.4f}  pix {val_pix:.4f}")

        if not args.no_wandb:
            wandb.log({
                f"seg_{mode}/train_loss": tr_loss,
                f"seg_{mode}/val_loss":   val_loss,
                f"seg_{mode}/dice":       val_dice,
                f"seg_{mode}/pix_acc":    val_pix,
                "epoch": epoch,
            })

        if val_dice > best_dice:
            best_dice = val_dice
            save_ckpt(model, epoch, best_dice, ckpt_path)

    if not args.no_wandb:
        wandb.finish()
    print(f"\nBest Dice [{mode}]: {best_dice:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task",       type=str, default="classification",
                   choices=["classification", "localization", "segmentation"])
    p.add_argument("--data_root",  type=str, default="/content/pets")
    p.add_argument("--ckpt_dir",   type=str,
                   default="/content/drive/MyDrive/DA6401_Assignment2/checkpoints")
    p.add_argument("--epochs",     type=int,   default=30)
    p.add_argument("--batch_size", type=int,   default=64)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--dropout_p",  type=float, default=0.4)
    p.add_argument("--unet_mode",  type=str,   default="full",
                   choices=["frozen", "partial", "full"])
    p.add_argument("--no_wandb",   action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.task == "classification":
        train_classifier(args)
    elif args.task == "localization":
        train_localizer(args)
    else:
        train_segmentation(args)
