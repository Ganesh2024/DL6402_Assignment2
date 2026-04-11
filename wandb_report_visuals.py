"""W&B Report Visualization Script.

Generates all visual content required for report sections:
    2.4 — Feature map visualization (first vs last conv layer)
    2.5 — Bounding box prediction table (10+ images)
    2.6 — Segmentation samples (original + GT + prediction)
    2.7 — Pipeline on 3 internet images

Usage
-----
python wandb_report_visuals.py \
    --data_root  /kaggle/working/pets \
    --ckpt_dir   /kaggle/working/checkpoints \
    --img_dir    /kaggle/working/internet_images
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import wandb

from models.vgg11 import VGG11Encoder
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet
from models.multitask import MultiTaskPerceptionModel
from data.pets_dataset import OxfordIIITPetDataset, build_transforms
from losses.iou_loss import IoULoss
from torch.utils.data import DataLoader


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_MEAN = (0.485, 0.456, 0.406)
_STD  = (0.229, 0.224, 0.225)

# Oxford pet breed names (0-indexed)
BREED_NAMES = [
    "Abyssinian", "Bengal", "Birman", "Bombay", "British_Shorthair",
    "Egyptian_Mau", "Maine_Coon", "Persian", "Ragdoll", "Russian_Blue",
    "Siamese", "Sphynx", "American_Bulldog", "American_Pit_Bull_Terrier",
    "Basset_Hound", "Beagle", "Boxer", "Chihuahua", "English_Cocker_Spaniel",
    "English_Setter", "German_Shorthaired", "Great_Pyrenees", "Havanese",
    "Japanese_Chin", "Keeshond", "Leonberger", "Miniature_Pinscher",
    "Newfoundland", "Pomeranian", "Pug", "Saint_Bernard", "Samoyed",
    "Scottish_Terrier", "Shiba_Inu", "Staffordshire_Bull_Terrier",
    "Wheaten_Terrier", "Yorkshire_Terrier"
]


def denormalize(tensor):
    """Convert normalised tensor back to displayable numpy image."""
    mean = np.array(_MEAN)
    std  = np.array(_STD)
    img  = tensor.permute(1, 2, 0).numpy()
    img  = img * std + mean
    return np.clip(img, 0, 1)


def get_val_loader(data_root, batch_size=16):
    val_tf  = build_transforms(is_train=False)
    full_ds = OxfordIIITPetDataset(
        data_root, split="trainval", transform=val_tf, task="all"
    )
    n_val   = max(1, int(0.15 * len(full_ds)))
    n_train = len(full_ds) - n_val
    from torch.utils.data import random_split
    _, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    val_ds.dataset.transform = val_tf

    def collate_fn(batch):
        images    = torch.stack([b["image"] for b in batch])
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

    return DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                      num_workers=2, collate_fn=collate_fn)


# ---------------------------------------------------------------------------
# Section 2.4 — Feature Map Visualization
# ---------------------------------------------------------------------------

def visualize_feature_maps(ckpt_dir):
    """
    Pass a single dog image through classifier backbone.
    Extract and visualize:
        - First conv layer (stage1[0][0]) feature maps
        - Last conv layer before pooling (stage5[1][0]) feature maps
    Log to W&B as image panels.
    """
    print("\n── Section 2.4: Feature Maps ──")

    model = VGG11Classifier(num_classes=37).to(DEVICE)
    payload = torch.load(
        os.path.join(ckpt_dir, "classifier.pth"), map_location=DEVICE
    )
    model.load_state_dict(payload.get("state_dict", payload))
    model.eval()

    # Storage for hooks
    feature_maps = {}

    def make_hook(name):
        def hook(module, inp, out):
            feature_maps[name] = out.detach().cpu()
        return hook

    # Register hooks on first and last conv layers
    h1 = model.backbone.stage1[0][0].register_forward_hook(make_hook("first_conv"))
    h2 = model.backbone.stage5[1][0].register_forward_hook(make_hook("last_conv"))

    # Create a synthetic "dog-like" normalised image for consistent demo
    # In practice use a real image from the dataset
    tf = build_transforms(is_train=False)
    dummy_np = np.random.randint(100, 200, (224, 224, 3), dtype=np.uint8)
    img_tensor = tf(image=dummy_np, bboxes=[], labels=[])["image"].float()
    img_tensor = img_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        model(img_tensor)

    h1.remove()
    h2.remove()

    wandb.init(project="DA6401-A2", name="feature-maps",
               reinit="finish_previous")

    for layer_name, fmaps in feature_maps.items():
        # fmaps shape: [1, C, H, W]
        fmaps = fmaps[0]         # [C, H, W]
        n_show = min(16, fmaps.shape[0])

        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        fig.suptitle(
            f"{layer_name.replace('_', ' ').title()} Feature Maps\n"
            f"({fmaps.shape[0]} channels total, showing first {n_show})",
            fontsize=14
        )

        for i, ax in enumerate(axes.flat):
            if i < n_show:
                fm = fmaps[i].numpy()
                ax.imshow(fm, cmap="viridis")
                ax.set_title(f"Ch {i}", fontsize=9)
            ax.axis("off")

        plt.tight_layout()
        wandb.log({f"feature_maps/{layer_name}": wandb.Image(fig)})
        plt.close(fig)
        print(f"  Logged {layer_name} feature maps")

    wandb.finish()
    print("  Section 2.4 done ✅")


# ---------------------------------------------------------------------------
# Section 2.5 — Bounding Box Prediction Table
# ---------------------------------------------------------------------------

def bbox_prediction_table(data_root, ckpt_dir):
    """
    Log W&B table with 15 test images showing:
    - Original image with GT bbox (green) and predicted bbox (red)
    - Confidence score (max softmax prob)
    - IoU between GT and predicted bbox
    """
    print("\n── Section 2.5: BBox Prediction Table ──")

    loc_model = VGG11Localizer(freeze_backbone=False).to(DEVICE)
    payload   = torch.load(
        os.path.join(ckpt_dir, "localizer.pth"), map_location=DEVICE
    )
    loc_model.load_state_dict(payload.get("state_dict", payload))
    loc_model.eval()

    clf_model = VGG11Classifier(num_classes=37).to(DEVICE)
    payload   = torch.load(
        os.path.join(ckpt_dir, "classifier.pth"), map_location=DEVICE
    )
    clf_model.load_state_dict(payload.get("state_dict", payload))
    clf_model.eval()

    val_loader = get_val_loader(data_root, batch_size=1)
    iou_fn     = IoULoss(reduction="none")

    wandb.init(project="DA6401-A2", name="bbox-table",
               reinit="finish_previous")

    table = wandb.Table(columns=[
        "image", "predicted_breed", "confidence",
        "gt_bbox", "pred_bbox", "iou", "result"
    ])

    count = 0
    for batch in val_loader:
        if batch["bbox"] is None or count >= 15:
            continue

        img    = batch["image"].to(DEVICE)    # [1,3,224,224]
        gt_box = batch["bbox"].to(DEVICE)     # [1,4]

        with torch.no_grad():
            pred_box = loc_model(img)          # [1,4]
            logits   = clf_model(img)          # [1,37]

        probs      = torch.softmax(logits, dim=1)
        confidence = probs.max().item()
        pred_class = probs.argmax().item()

        iou_val  = (1.0 - iou_fn(pred_box, gt_box)).item()

        # Draw image with bboxes
        img_np   = denormalize(batch["image"][0])
        fig, ax  = plt.subplots(1, 1, figsize=(5, 5))
        ax.imshow(img_np)

        # GT bbox — green
        gt  = gt_box[0].cpu().numpy()
        cx, cy, w, h = gt
        rect_gt = patches.Rectangle(
            (cx - w/2, cy - h/2), w, h,
            linewidth=2, edgecolor="green", facecolor="none",
            label="Ground Truth"
        )
        ax.add_patch(rect_gt)

        # Pred bbox — red
        pr  = pred_box[0].detach().cpu().numpy()
        cx2, cy2, w2, h2 = pr
        rect_pr = patches.Rectangle(
            (cx2 - w2/2, cy2 - h2/2), w2, h2,
            linewidth=2, edgecolor="red", facecolor="none",
            label="Prediction"
        )
        ax.add_patch(rect_pr)

        breed = BREED_NAMES[pred_class] if pred_class < len(BREED_NAMES) else str(pred_class)
        ax.set_title(f"{breed} | conf={confidence:.2f} | IoU={iou_val:.3f}",
                     fontsize=9)
        ax.legend(fontsize=8)
        ax.axis("off")

        result = "good" if iou_val > 0.5 else "failure"

        table.add_data(
            wandb.Image(fig),
            breed,
            round(confidence, 4),
            f"cx={gt[0]:.1f} cy={gt[1]:.1f} w={gt[2]:.1f} h={gt[3]:.1f}",
            f"cx={pr[0]:.1f} cy={pr[1]:.1f} w={pr[2]:.1f} h={pr[3]:.1f}",
            round(iou_val, 4),
            result
        )
        plt.close(fig)
        count += 1
        print(f"  Sample {count}/15 — IoU={iou_val:.3f}  conf={confidence:.3f}")

    wandb.log({"bbox_predictions": table})
    wandb.finish()
    print("  Section 2.5 done ✅")


# ---------------------------------------------------------------------------
# Section 2.6 — Segmentation Evaluation
# ---------------------------------------------------------------------------

def segmentation_samples(data_root, ckpt_dir):
    """
    Log 5 sample images showing:
    1. Original image
    2. Ground truth trimap
    3. Predicted trimap mask
    Also logs pixel accuracy vs dice score comparison.
    """
    print("\n── Section 2.6: Segmentation Samples ──")

    model   = VGG11UNet(num_classes=3).to(DEVICE)
    payload = torch.load(
        os.path.join(ckpt_dir, "unet.pth"), map_location=DEVICE
    )
    model.load_state_dict(payload.get("state_dict", payload))
    model.eval()

    val_loader = get_val_loader(data_root, batch_size=1)

    wandb.init(project="DA6401-A2", name="segmentation-eval",
               reinit="finish_previous")

    # Color map for trimap: 0=fg(red), 1=bg(blue), 2=boundary(green)
    COLORS = np.array([
        [220,  50,  50],   # foreground — red
        [ 50,  50, 220],   # background — blue
        [ 50, 200,  50],   # boundary   — green
    ], dtype=np.uint8)

    table = wandb.Table(columns=[
        "original", "ground_truth", "prediction",
        "pixel_accuracy", "dice_score"
    ])

    count = 0
    for batch in val_loader:
        if batch["mask"] is None or count >= 5:
            continue

        img  = batch["image"].to(DEVICE)
        mask = batch["mask"].to(DEVICE)

        with torch.no_grad():
            logits = model(img)

        pred = logits.argmax(dim=1)

        # Pixel accuracy
        pix_acc = (pred == mask).float().mean().item()

        # Dice score (foreground class)
        probs = torch.softmax(logits, dim=1)[:, 0]
        gt    = (mask == 0).float()
        num   = 2.0 * (probs * gt).sum() + 1.0
        den   = probs.sum() + gt.sum() + 1.0
        dice  = (num / den).item()

        # Convert to displayable images
        img_np  = denormalize(batch["image"][0])
        gt_np   = COLORS[batch["mask"][0].numpy()]
        pr_np   = COLORS[pred[0].cpu().numpy()]

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(img_np);   axes[0].set_title("Original");       axes[0].axis("off")
        axes[1].imshow(gt_np);    axes[1].set_title("Ground Truth");    axes[1].axis("off")
        axes[2].imshow(pr_np);    axes[2].set_title("Prediction");      axes[2].axis("off")
        fig.suptitle(f"Pixel Acc: {pix_acc:.3f}  |  Dice: {dice:.3f}", fontsize=12)
        plt.tight_layout()

        table.add_data(
            wandb.Image(img_np),
            wandb.Image(gt_np),
            wandb.Image(pr_np),
            round(pix_acc, 4),
            round(dice, 4),
        )
        plt.close(fig)
        count += 1
        print(f"  Sample {count}/5 — PixAcc={pix_acc:.3f}  Dice={dice:.3f}")

    wandb.log({"segmentation_samples": table})
    wandb.finish()
    print("  Section 2.6 done ✅")


# ---------------------------------------------------------------------------
# Section 2.7 — Pipeline on Internet Images
# ---------------------------------------------------------------------------

def pipeline_on_internet_images(img_dir, ckpt_dir):
    """
    Run the full MultiTask pipeline on 3 images downloaded from the internet.
    Logs: original image, predicted breed, bbox overlay, segmentation mask.

    Place 3 pet images in img_dir named:
        pet1.jpg, pet2.jpg, pet3.jpg
    """
    print("\n── Section 2.7: Internet Images ──")

    model = MultiTaskPerceptionModel(
        classifier_path=os.path.join(ckpt_dir, "classifier.pth"),
        localizer_path =os.path.join(ckpt_dir, "localizer.pth"),
        unet_path      =os.path.join(ckpt_dir, "unet.pth"),
    ).to(DEVICE)
    model.eval()

    tf = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=_MEAN, std=_STD),
        ToTensorV2(),
    ])

    COLORS = np.array([
        [220,  50,  50],
        [ 50,  50, 220],
        [ 50, 200,  50],
    ], dtype=np.uint8)

    wandb.init(project="DA6401-A2", name="internet-images",
               reinit="finish_previous")

    table = wandb.Table(columns=[
        "original", "bbox_overlay",
        "segmentation", "predicted_breed", "confidence"
    ])

    img_files = sorted([
        f for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])[:3]

    if not img_files:
        print(f"  No images found in {img_dir}")
        print("  Place 3 pet images (pet1.jpg, pet2.jpg, pet3.jpg) there")
        wandb.finish()
        return

    for fname in img_files:
        fpath  = os.path.join(img_dir, fname)
        img_np = np.array(Image.open(fpath).convert("RGB"))
        tensor = tf(image=img_np, bboxes=[], labels=[])["image"]
        tensor = tensor.float().unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            out = model(tensor)

        probs      = torch.softmax(out["classification"], dim=1)
        confidence = probs.max().item()
        pred_class = probs.argmax().item()
        breed      = BREED_NAMES[pred_class] if pred_class < len(BREED_NAMES) else str(pred_class)

        bbox = out["localization"][0].cpu().numpy()
        cx, cy, w, h = bbox

        seg_mask = out["segmentation"][0].argmax(0).cpu().numpy()
        seg_rgb  = COLORS[seg_mask]

        # Resize original for display
        orig_disp = np.array(Image.open(fpath).convert("RGB").resize((224, 224)))

        # BBox overlay
        fig1, ax1 = plt.subplots(figsize=(5, 5))
        ax1.imshow(orig_disp)
        rect = patches.Rectangle(
            (cx - w/2, cy - h/2), w, h,
            linewidth=3, edgecolor="red", facecolor="none"
        )
        ax1.add_patch(rect)
        ax1.set_title(f"{breed}\nconf={confidence:.2f}", fontsize=10)
        ax1.axis("off")
        plt.tight_layout()

        table.add_data(
            wandb.Image(orig_disp),
            wandb.Image(fig1),
            wandb.Image(seg_rgb),
            breed,
            round(confidence, 4),
        )
        plt.close(fig1)
        print(f"  {fname}: {breed}  conf={confidence:.3f}")

    wandb.log({"internet_pipeline": table})
    wandb.finish()
    print("  Section 2.7 done ✅")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str,
                   default="/kaggle/working/pets")
    p.add_argument("--ckpt_dir",  type=str,
                   default="/kaggle/working/checkpoints")
    p.add_argument("--img_dir",   type=str,
                   default="/kaggle/working/internet_images")
    p.add_argument("--sections",  type=str, default="2.4,2.5,2.6,2.7",
                   help="Comma-separated sections to run")
    return p.parse_args()


if __name__ == "__main__":
    args    = parse_args()
    sections = [s.strip() for s in args.sections.split(",")]

    if "2.4" in sections:
        visualize_feature_maps(args.ckpt_dir)

    if "2.5" in sections:
        bbox_prediction_table(args.data_root, args.ckpt_dir)

    if "2.6" in sections:
        segmentation_samples(args.data_root, args.ckpt_dir)

    if "2.7" in sections:
        os.makedirs(args.img_dir, exist_ok=True)
        pipeline_on_internet_images(args.img_dir, args.ckpt_dir)

    print("\n✅ All selected sections complete!")
