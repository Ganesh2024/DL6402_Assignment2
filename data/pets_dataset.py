"""Oxford-IIIT Pet Dataset loader.

Handles all three annotation types:
    * class label  (37 breed classes, 0-indexed)
    * bounding box (head, pixel space → cx, cy, w, h)
    * trimap mask  (1=fg, 2=bg, 3=boundary → remapped to 0,1,2)

Expected on-disk layout:
    <root>/
        images/
        annotations/
            trimaps/
            xmls/
            list.txt
            trainval.txt
            test.txt
"""

import os
import xml.etree.ElementTree as ET
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2


_MEAN   = (0.485, 0.456, 0.406)
_STD    = (0.229, 0.224, 0.225)
IMG_DIM = 224


def build_transforms(is_train: bool = True) -> A.Compose:
    """
    Training  → aggressive augmentation to fight overfitting.
    Validation → resize + normalize only.
    """
    bbox_params = A.BboxParams(
        format="coco",
        label_fields=["labels"],
        min_visibility=0.2,
    )
    if is_train:
        ops = [
            A.Resize(IMG_DIM, IMG_DIM),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.RandomRotate90(p=0.2),
            A.ColorJitter(brightness=0.3, contrast=0.3,
                          saturation=0.3, hue=0.1, p=0.6),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.GaussNoise(p=0.2),
            A.RandomGamma(p=0.2),
            A.Affine(scale=(0.85, 1.15), translate_percent=0.05,
                     rotate=(-15, 15), p=0.5),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
            A.Normalize(mean=_MEAN, std=_STD),
            ToTensorV2(),
        ]
    else:
        ops = [
            A.Resize(IMG_DIM, IMG_DIM),
            A.Normalize(mean=_MEAN, std=_STD),
            ToTensorV2(),
        ]
    return A.Compose(ops, bbox_params=bbox_params)


class OxfordIIITPetDataset(Dataset):
    """
    Multi-task dataset for Oxford-IIIT Pet benchmark.

    Args:
        root      : path to dataset root directory
        split     : 'trainval' | 'test'
        transform : albumentations Compose pipeline
        task      : 'classification' | 'localization' | 'segmentation' | 'all'
    """

    def __init__(
        self,
        root: str,
        split: str = "trainval",
        transform: Optional[A.Compose] = None,
        task: str = "all",
    ):
        self.root      = root
        self.split     = split
        self.task      = task
        self.transform = (
            transform if transform is not None
            else build_transforms(is_train=(split == "trainval"))
        )
        self.img_dir   = os.path.join(root, "images")
        self.mask_dir  = os.path.join(root, "annotations", "trimaps")
        self.xml_dir   = os.path.join(root, "annotations", "xmls")
        self.list_path = os.path.join(root, "annotations", f"{split}.txt")

        self.samples: List[Tuple[str, int]] = []
        self._load_split()

    def _load_split(self):
        with open(self.list_path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts    = line.split()
                stem     = parts[0]
                class_id = int(parts[1]) - 1

                # skip missing images
                if not os.path.exists(
                    os.path.join(self.img_dir, f"{stem}.jpg")
                ):
                    continue

                # for localization task — skip samples with no bbox XML
                if self.task == "localization":
                    xml_path = os.path.join(self.xml_dir, f"{stem}.xml")
                    if not os.path.exists(xml_path):
                        continue
                    # also skip zero-area boxes
                    bbox = self._parse_xml_bbox(xml_path,
                                                224, 224)
                    if bbox is None:
                        continue
                    w, h = bbox[2], bbox[3]
                    if w < 1.0 or h < 1.0:
                        continue

                self.samples.append((stem, class_id))

    @staticmethod
    def _parse_xml_bbox(xml_path: str, img_w: int, img_h: int):
        if not os.path.exists(xml_path):
            return None
        try:
            root = ET.parse(xml_path).getroot()
            obj  = root.find("object")
            if obj is None:
                return None
            bb   = obj.find("bndbox")
            sz   = root.find("size")
            ow   = float(sz.find("width").text)
            oh   = float(sz.find("height").text)
            x1   = float(bb.find("xmin").text) / ow * img_w
            y1   = float(bb.find("ymin").text) / oh * img_h
            x2   = float(bb.find("xmax").text) / ow * img_w
            y2   = float(bb.find("ymax").text) / oh * img_h
            return [(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1]
        except Exception:
            return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        stem, class_id = self.samples[idx]

        img_arr = np.array(
            Image.open(os.path.join(self.img_dir, f"{stem}.jpg")).convert("RGB")
        )
        h, w = img_arr.shape[:2]

        mask_arr = None
        mp = os.path.join(self.mask_dir, f"{stem}.png")
        if os.path.exists(mp):
            mask_arr = (np.array(Image.open(mp)) - 1).astype(np.int64)

        raw_bbox = self._parse_xml_bbox(
            os.path.join(self.xml_dir, f"{stem}.xml"), w, h
        )

        albu_boxes, albu_labels = [], []
        if raw_bbox is not None:
            cx, cy, bw, bh = raw_bbox
            albu_boxes  = [[cx - bw/2, cy - bh/2, bw, bh]]
            albu_labels = [0]

        aug_in = {"image": img_arr, "bboxes": albu_boxes, "labels": albu_labels}
        if mask_arr is not None:
            aug_in["mask"] = mask_arr

        out = self.transform(**aug_in)

        img_tensor  = out["image"].float()

        mask_tensor = None
        if mask_arr is not None:
            m = out["mask"]
            mask_tensor = (
                m.long() if isinstance(m, torch.Tensor)
                else torch.from_numpy(np.array(m)).long()
            )

        bbox_tensor = None
        if out["bboxes"]:
            xmin, ymin, bw, bh = out["bboxes"][0]
            bbox_tensor = torch.tensor(
                [xmin + bw/2, ymin + bh/2, bw, bh],
                dtype=torch.float32
            )

        return {
            "image":    img_tensor,
            "class_id": torch.tensor(class_id, dtype=torch.long),
            "bbox":     bbox_tensor,
            "mask":     mask_tensor,
            "stem":     stem,
        }
