"""VGG11-based breed classifier — trained entirely from scratch.

Design rationale
----------------
BN placement
    BatchNorm is inserted before ReLU in each FC block. Pre-activation BN
    keeps the input distribution to each layer centred and unit-scaled,
    which stabilises gradients and allows a higher learning rate.

CustomDropout placement
    Dropout is placed after BN + ReLU. Applying dropout on already-normalised
    activations means BN running statistics are computed on the full feature
    set rather than a partially zeroed one, giving more stable BN estimates
    especially in the first few epochs.

FC head size
    4096 → 4096 → 37 mirrors the original VGG paper's classifier exactly.
    This large capacity is appropriate given the depth of features produced
    by the backbone (512 × 7 × 7 = 25 088 input dimensions).
"""

import torch
import torch.nn as nn

from .vgg11 import VGG11Encoder
from .layers import CustomDropout


class VGG11Classifier(nn.Module):
    """Complete classification model: VGG11 backbone + 3-layer FC head."""

    def __init__(
        self,
        num_classes: int = 37,
        in_channels: int = 3,
        dropout_p: float = 0.5,
    ):
        super().__init__()
        self.backbone = VGG11Encoder(in_channels=in_channels)

        flat_dim = 512 * 7 * 7   # 25088

        self.fc_head = nn.Sequential(
            nn.Linear(flat_dim, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),

            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),

            nn.Linear(4096, num_classes),
        )

        self._init_fc()

    def _init_fc(self):
        for m in self.fc_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        flat = feat.view(feat.size(0), -1)
        return self.fc_head(flat)
