"""Bounding-box localisation model.

The VGG11 convolutional backbone is reused as a frozen (or optionally
fine-tuned) feature extractor and a small regression head is attached to
predict the head bounding box in pixel space.

Output format: [x_center, y_center, width, height]  (all in pixels, 224×224)

Backbone freezing decision
--------------------------
By default the backbone weights are *frozen* during localisation training.
Rationale: the classifier pre-training already teaches the backbone to detect
pet-relevant spatial features (snouts, eyes, fur texture).  Freezing keeps
those representations intact and lets the regression head train quickly
without catastrophic forgetting.  If you observe the regression plateauing
early, switch `freeze_backbone=False` to allow end-to-end fine-tuning.

The regression head uses a Sigmoid gating trick to keep outputs in [0, 224]:
    output = sigmoid(raw) * 224
This prevents the network from predicting boxes way outside the image boundary
and also gives the loss function numerically well-behaved gradients.
"""

import torch
import torch.nn as nn

from .vgg11 import VGG11Encoder
from .layers import CustomDropout


class VGG11Localizer(nn.Module):
    """VGG11 encoder + 4-dimensional regression head."""

    def __init__(
        self,
        in_channels: int = 3,
        dropout_p: float = 0.5,
        freeze_backbone: bool = True,
        img_size: int = 224,
    ):
        super().__init__()
        self.img_size = img_size
        self.backbone  = VGG11Encoder(in_channels=in_channels)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        flat_dim = 512 * 7 * 7

        # Regression head: 25088 → 1024 → 256 → 4
        self.reg_head = nn.Sequential(
            nn.Linear(flat_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),

            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),

            nn.Linear(256, 4),   # raw, unbounded logits
        )

        self._init_head()

    # ------------------------------------------------------------------
    def _init_head(self):
        for m in self.reg_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, in_channels, 224, 224] normalised image batch

        Returns:
            bbox [B, 4]  (x_center, y_center, width, height) in pixels
        """
        feat  = self.backbone(x)                        # [B, 512, 7, 7]
        flat  = feat.view(feat.size(0), -1)             # [B, 25088]
        raw   = self.reg_head(flat)                     # [B, 4]

        # Map to pixel space [0, img_size] via sigmoid scaling
        bbox  = torch.sigmoid(raw) * self.img_size      # [B, 4]
        return bbox
