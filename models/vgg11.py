"""VGG11 convolutional encoder.

Architecture follows Table 1 (column A) of:
    Simonyan & Zisserman, "Very Deep Convolutional Networks for Large-Scale
    Image Recognition", ICLR 2015  (arXiv:1409.1556).

Pretrained weights
------------------
When pretrained=True, official VGG11-BN weights from torchvision are mapped
into our custom stage-based architecture.  This gives the encoder a strong
ImageNet initialisation so classifier/segmentation heads converge quickly.
torchvision ships with torch so no extra import is needed.
"""

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn

from .layers import CustomDropout


def _conv_bn_relu(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class VGG11Encoder(nn.Module):
    """
    VGG-11 convolutional backbone (5 pooling stages).

    Stage output shapes for 224×224 input:
        stage1 →  64 @ 112×112
        stage2 → 128 @  56×56
        stage3 → 256 @  28×28
        stage4 → 512 @  14×14
        stage5 → 512 @   7×7
    """

    def __init__(self, in_channels: int = 3, pretrained: bool = False):
        super().__init__()

        self.stage1 = nn.Sequential(
            _conv_bn_relu(in_channels, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.stage2 = nn.Sequential(
            _conv_bn_relu(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.stage3 = nn.Sequential(
            _conv_bn_relu(128, 256),
            _conv_bn_relu(256, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.stage4 = nn.Sequential(
            _conv_bn_relu(256, 512),
            _conv_bn_relu(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.stage5 = nn.Sequential(
            _conv_bn_relu(512, 512),
            _conv_bn_relu(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))

        if pretrained:
            self._load_pretrained_weights()
        else:
            self._kaiming_init()

    def _kaiming_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _load_pretrained_weights(self):
        """
        Map torchvision VGG11-BN pretrained weights into our stage layout.

        torchvision features layer indices (VGG11_BN):
            0-2  : Conv(3→64),   BN, ReLU   → stage1[0]
            3    : MaxPool                   → stage1[1]
            4-6  : Conv(64→128), BN, ReLU   → stage2[0]
            7    : MaxPool                   → stage2[1]
            8-10 : Conv(128→256),BN, ReLU   → stage3[0]
            11-13: Conv(256→256),BN, ReLU   → stage3[1]
            14   : MaxPool                   → stage3[2]
            15-17: Conv(256→512),BN, ReLU   → stage4[0]
            18-20: Conv(512→512),BN, ReLU   → stage4[1]
            21   : MaxPool                   → stage4[2]
            22-24: Conv(512→512),BN, ReLU   → stage5[0]
            25-27: Conv(512→512),BN, ReLU   → stage5[1]
            28   : MaxPool                   → stage5[2]
        """
        try:
            import torchvision.models as tvm
            ref = tvm.vgg11_bn(
                weights=tvm.VGG11_BN_Weights.IMAGENET1K_V1
            )
            src = ref.features

            def _copy(dst_block, conv_idx):
                # dst_block = _conv_bn_relu Sequential [Conv2d, BN, ReLU]
                dst_block[0].weight.data.copy_(src[conv_idx].weight.data)
                bn = src[conv_idx + 1]
                dst_block[1].weight.data.copy_(bn.weight.data)
                dst_block[1].bias.data.copy_(bn.bias.data)
                dst_block[1].running_mean.copy_(bn.running_mean)
                dst_block[1].running_var.copy_(bn.running_var)

            _copy(self.stage1[0], 0)
            _copy(self.stage2[0], 4)
            _copy(self.stage3[0], 8)
            _copy(self.stage3[1], 11)
            _copy(self.stage4[0], 15)
            _copy(self.stage4[1], 18)
            _copy(self.stage5[0], 22)
            _copy(self.stage5[1], 25)

            print("  → Pretrained ImageNet weights loaded into VGG11Encoder")
            del ref

        except Exception as exc:
            print(f"  ⚠ Pretrained load failed ({exc}). Using Kaiming init.")
            self._kaiming_init()

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        s1 = self.stage1(x)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)
        s5 = self.stage5(s4)
        bottleneck = self.adaptive_pool(s5)

        if return_features:
            return bottleneck, {
                "s1": s1, "s2": s2, "s3": s3, "s4": s4, "s5": s5
            }
        return bottleneck
