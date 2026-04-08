"""Inference utilities — run the unified pipeline on arbitrary images."""

import numpy as np
import torch
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from multitask import MultiTaskPerceptionModel

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)
INPUT_SIZE     = 224

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess(img_path: str) -> torch.Tensor:
    """Load an image from disk and return a normalised [1,3,224,224] tensor."""
    tf = A.Compose([
        A.Resize(INPUT_SIZE, INPUT_SIZE),
        A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ToTensorV2(),
    ])
    raw = np.array(Image.open(img_path).convert("RGB"))
    return tf(image=raw)["image"].unsqueeze(0).float()


def run_inference(img_path: str) -> dict:
    """
    Run the full multi-task pipeline on a single image.

    Returns a dict with:
        'class_idx'   : predicted breed index (int)
        'bbox'        : [cx, cy, w, h] in pixels
        'seg_mask'    : [224, 224] predicted class map (0=fg,1=bg,2=boundary)
    """
    model = MultiTaskPerceptionModel().to(DEVICE)
    model.eval()

    img_tensor = preprocess(img_path).to(DEVICE)
    with torch.no_grad():
        outputs = model(img_tensor)

    class_idx = outputs["classification"].argmax(dim=1).item()
    bbox      = outputs["localization"][0].cpu().numpy().tolist()
    seg_mask  = outputs["segmentation"][0].argmax(dim=0).cpu().numpy()

    return {"class_idx": class_idx, "bbox": bbox, "seg_mask": seg_mask}


if __name__ == "__main__":
    import sys
    result = run_inference(sys.argv[1])
    print(f"Predicted breed index : {result['class_idx']}")
    print(f"Bounding box (cx,cy,w,h): {result['bbox']}")
    print(f"Segmentation mask shape : {result['seg_mask'].shape}")
