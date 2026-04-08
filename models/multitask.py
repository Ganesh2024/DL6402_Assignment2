"""Unified multi-task perception model.

Loads weights from three separately-trained checkpoints:
    checkpoints/classifier.pth
    checkpoints/localizer.pth
    checkpoints/unet.pth

A single forward pass through the shared VGG11 backbone simultaneously
produces breed-classification logits, a bounding-box prediction, and a
pixel-wise segmentation map.

Checkpoint format (either of the two is accepted):
    • plain state_dict
    • dict with key 'state_dict'
"""

import os

import torch
import torch.nn as nn

from models.vgg11 import VGG11Encoder
from models.layers import CustomDropout
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet


# ---------------------------------------------------------------------------
def _load_state(path: str, device: torch.device) -> dict:
    """Load a .pth file and return the raw state_dict regardless of format."""
    payload = torch.load(path, map_location=device)
    if isinstance(payload, dict) and "state_dict" in payload:
        return payload["state_dict"]
    return payload


# ---------------------------------------------------------------------------
class MultiTaskPerceptionModel(nn.Module):
    """
    Shared-backbone multi-task model.

    The backbone is taken from the *classifier* checkpoint (Task 1) since it
    was trained for the longest time and captures the richest feature set.
    The localisation and segmentation heads are then sourced from their own
    checkpoints.

    All paths are relative to the repository root (where multitask.py lives).
    """

    def __init__(
        self,
        num_breeds: int = 37,
        seg_classes: int = 3,
        in_channels: int = 3,
        classifier_path: str = "checkpoints/classifier.pth",
        localizer_path: str = "checkpoints/localizer.pth",
        unet_path: str = "checkpoints/unet.pth",
    ):
        import gdown
        gdown.download(id="1rO3Cxf4sJppGtda6wr82cAG1B5BnimtD", output=classifier_path, quiet=False)
        gdown.download(id="1gU2kBEb-XgX2o_F5hRX-fUSCnKELkKCD", output=localizer_path, quiet=False)
        gdown.download(id="1lN2CT1Nr8gwO42SqGpje7iLf-QIwp_eS", output=unet_path, quiet=False)
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ------------------------https://drive.google.com/file/d/1gU2kBEb-XgX2o_F5hRX-fUSCnKELkKCD/view?usp=sharing-----------------------https://drive.google.com/file/d//view?usp=sharing------------------- https://drive.google.com/file/d//view?usp=sharing
        # 1. Build full models (so we have the right architecture to load into)
        # ------------------------------------------------------------------
        clf_model  = VGG11Classifier(num_classes=num_breeds, in_channels=in_channels)
        loc_model  = VGG11Localizer(in_channels=in_channels)
        seg_model  = VGG11UNet(num_classes=seg_classes, in_channels=in_channels)

        # ------------------------------------------------------------------
        # 2. Load saved weights
        # ------------------------------------------------------------------
        clf_model.load_state_dict(_load_state(classifier_path, device))
        loc_model.load_state_dict(_load_state(localizer_path,  device))
        seg_model.load_state_dict(_load_state(unet_path,       device))

        # ------------------------------------------------------------------
        # 3. Shared backbone — taken from the classifier (richest features)
        # ------------------------------------------------------------------
        self.backbone = clf_model.backbone

        # ------------------------------------------------------------------
        # 4. Task heads (everything except the backbone)
        # ------------------------------------------------------------------
        self.clf_head = clf_model.fc_head

        # Localiser head = reg_head only (backbone already shared above)
        self.loc_head = loc_model.reg_head
        self.img_size = loc_model.img_size

        # Segmentation decoder (all decoder layers + output_conv)
        self.seg_decoder = nn.ModuleDict({
            "up5":        seg_model.up5,
            "up4":        seg_model.up4,
            "up3":        seg_model.up3,
            "up2":        seg_model.up2,
            "up1":        seg_model.up1,
            "dec5":       seg_model.dec5,
            "dec4":       seg_model.dec4,
            "dec3":       seg_model.dec3,
            "dec2":       seg_model.dec2,
            "dec1":       seg_model.dec1,
            "output_conv": seg_model.output_conv,
            "drop":       seg_model.drop,
        })

        self.to(device)

    # ------------------------------------------------------------------
    def _decode_segmentation(self, bottleneck, skips):
        """Run the segmentation decoder path given encoder outputs."""
        sd = self.seg_decoder
        drop = sd["drop"]

        d5 = sd["up5"](bottleneck)
        d5 = torch.cat([d5, skips["s4"]], dim=1)
        d5 = drop(sd["dec5"](d5))

        d4 = sd["up4"](d5)
        d4 = torch.cat([d4, skips["s3"]], dim=1)
        d4 = drop(sd["dec4"](d4))

        d3 = sd["up3"](d4)
        d3 = torch.cat([d3, skips["s2"]], dim=1)
        d3 = drop(sd["dec3"](d3))

        d2 = sd["up2"](d3)
        d2 = torch.cat([d2, skips["s1"]], dim=1)
        d2 = drop(sd["dec2"](d2))

        d1 = sd["up1"](d2)
        d1 = sd["dec1"](d1)

        return sd["output_conv"](d1)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> dict:
        """
        Single forward pass through the shared backbone.

        Args:
            x: [B, 3, 224, 224] normalised image batch

        Returns:
            dict with keys:
              'classification' : [B, num_breeds]         – breed logits
              'localization'   : [B, 4]                  – (cx,cy,w,h) pixels
              'segmentation'   : [B, seg_classes, 224, 224] – mask logits
        """
        # One backbone call produces both the bottleneck and skip maps
        bottleneck, skips = self.backbone(x, return_features=True)

        # --- Classification head ---
        flat   = bottleneck.view(bottleneck.size(0), -1)
        cls_out = self.clf_head(flat)

        # --- Localisation head ---
        raw_bbox = self.loc_head(flat)
        loc_out  = torch.sigmoid(raw_bbox) * self.img_size

        # --- Segmentation head ---
        seg_out = self._decode_segmentation(bottleneck, skips)

        return {
            "classification": cls_out,
            "localization":   loc_out,
            "segmentation":   seg_out,
        }
