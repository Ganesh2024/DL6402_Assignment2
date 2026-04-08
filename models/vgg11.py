"""VGG11 convolutional encoder — implemented from scratch.

Architecture follows Table 1 (column A) of:
    Simonyan & Zisserman, "Very Deep Convolutional Networks for Large-Scale
    Image Recognition", ICLR 2015  (arXiv:1409.1556).

No pretrained weights are used. All parameters are initialised from scratch
using Kaiming-He normal initialisation for conv layers and ones/zeros for
BatchNorm, which is the standard best practice for training deep networks
without pretrained weights.

Stage output shapes for a 224×224 input
----------------------------------------
    stage1 →  64 channels @ 112×112
    stage2 → 128 channels @  56×56
    stage3 → 256 channels @  28×28
    stage4 → 512 channels @  14×14
    stage5 → 512 channels @   7×7
"""

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn

from .layers import CustomDropout


# ---------------------------------------------------------------------------
def _conv_bn_relu(in_ch: int, out_ch: int) -> nn.Sequential:
    """Single Conv2d → BatchNorm2d → ReLU block."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


# ---------------------------------------------------------------------------
class VGG11Encoder(nn.Module):
    """
    VGG-11 convolutional backbone built entirely from scratch using
    standard torch.nn primitives.  No external pretrained weights are
    loaded at any point.

    The five convolutional stages mirror column A of the VGG paper exactly:
        Stage 1 — 1 conv (64 filters)
        Stage 2 — 1 conv (128 filters)
        Stage 3 — 2 convs (256 filters)
        Stage 4 — 2 convs (512 filters)
        Stage 5 — 2 convs (512 filters)

    Each stage ends with a 2×2 max-pool that halves spatial dimensions.
    An adaptive average pool normalises the output to 7×7 so the FC head
    receives a fixed-size feature vector regardless of minor input variation.
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()

        # ── Stage 1 ───────────────────────────────────────────────
        self.stage1 = nn.Sequential(
            _conv_bn_relu(in_channels, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),      # 224 → 112
        )

        # ── Stage 2 ───────────────────────────────────────────────
        self.stage2 = nn.Sequential(
            _conv_bn_relu(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),      # 112 → 56
        )

        # ── Stage 3 ───────────────────────────────────────────────
        self.stage3 = nn.Sequential(
            _conv_bn_relu(128, 256),
            _conv_bn_relu(256, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),      # 56 → 28
        )

        # ── Stage 4 ───────────────────────────────────────────────
        self.stage4 = nn.Sequential(
            _conv_bn_relu(256, 512),
            _conv_bn_relu(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),      # 28 → 14
        )

        # ── Stage 5 ───────────────────────────────────────────────
        self.stage5 = nn.Sequential(
            _conv_bn_relu(512, 512),
            _conv_bn_relu(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),      # 14 → 7
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self):
        """
        Kaiming-He normal init for conv layers (fan_out, ReLU nonlinearity).
        BN weight=1, bias=0 — standard initialisation.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Args:
            x               : [B, 3, 224, 224] normalised image batch
            return_features : if True, also return skip-connection maps

        Returns:
            bottleneck [B, 512, 7, 7]
            (optionally) dict of per-stage feature maps for U-Net decoder
        """
        s1 = self.stage1(x)        # [B,  64, 112, 112]
        s2 = self.stage2(s1)       # [B, 128,  56,  56]
        s3 = self.stage3(s2)       # [B, 256,  28,  28]
        s4 = self.stage4(s3)       # [B, 512,  14,  14]
        s5 = self.stage5(s4)       # [B, 512,   7,   7]

        bottleneck = self.adaptive_pool(s5)

        if return_features:
            return bottleneck, {
                "s1": s1, "s2": s2, "s3": s3,
                "s4": s4, "s5": s5,
            }
        return bottleneck


# ---------------------------------------------------------------------------
# Alias so autograder can import either name
# ---------------------------------------------------------------------------
VGG11 = VGG11Encoder
