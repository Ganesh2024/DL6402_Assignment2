"""VGG11-based breed classifier.

Design rationale
----------------
* Pretrained backbone — ImageNet VGG11-BN weights initialise the encoder;
  only the FC head needs to learn breed-specific features from scratch.
* Two-phase training (in train.py) — freeze backbone for warm-up epochs,
  then unfreeze for end-to-end fine-tuning with a lower backbone LR.
* BatchNorm before ReLU in FC blocks — stabilises activations, allows
  higher learning rates, speeds up convergence.
* CustomDropout after BN+ReLU — applied on normalised activations so BN
  running stats are computed on the full un-dropped feature set.
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
        pretrained: bool = True,
    ):
        super().__init__()
        self.backbone = VGG11Encoder(
            in_channels=in_channels,
            pretrained=pretrained,
        )
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
                nn.init.normal_(m.weight, 0.0, 0.01)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        flat = feat.view(feat.size(0), -1)
        return self.fc_head(flat)
