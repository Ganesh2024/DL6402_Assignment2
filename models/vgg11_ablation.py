import torch
import torch.nn as nn
from .layers import CustomDropout


def _conv_relu(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True),
        nn.ReLU(inplace=True),
    )


def _conv_bn_relu(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class VGG11EncoderAblation(nn.Module):
    def __init__(self, in_channels=3, use_bn=True):
        super().__init__()
        self.use_bn = use_bn
        block = _conv_bn_relu if use_bn else _conv_relu

        self.stage1 = nn.Sequential(block(in_channels, 64),  nn.MaxPool2d(2, 2))
        self.stage2 = nn.Sequential(block(64, 128),           nn.MaxPool2d(2, 2))
        self.stage3 = nn.Sequential(block(128, 256), block(256, 256), nn.MaxPool2d(2, 2))
        self.stage4 = nn.Sequential(block(256, 512), block(512, 512), nn.MaxPool2d(2, 2))
        self.stage5 = nn.Sequential(block(512, 512), block(512, 512), nn.MaxPool2d(2, 2))
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, return_features=False):
        s1 = self.stage1(x)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)
        s5 = self.stage5(s4)
        bottleneck = self.adaptive_pool(s5)
        if return_features:
            return bottleneck, {"s1": s1, "s2": s2, "s3": s3, "s4": s4, "s5": s5}
        return bottleneck


class VGG11ClassifierAblation(nn.Module):
    def __init__(self, num_classes=37, in_channels=3, dropout_p=0.5, use_bn=True):
        super().__init__()
        self.backbone = VGG11EncoderAblation(in_channels=in_channels, use_bn=use_bn)
        flat_dim = 512 * 7 * 7
        layers = []
        layers.append(nn.Linear(flat_dim, 4096))
        if use_bn:
            layers.append(nn.BatchNorm1d(4096))
        layers.append(nn.ReLU(inplace=True))
        if dropout_p > 0:
            layers.append(CustomDropout(p=dropout_p))
        layers.append(nn.Linear(4096, 4096))
        if use_bn:
            layers.append(nn.BatchNorm1d(4096))
        layers.append(nn.ReLU(inplace=True))
        if dropout_p > 0:
            layers.append(CustomDropout(p=dropout_p))
        layers.append(nn.Linear(4096, num_classes))
        self.fc_head = nn.Sequential(*layers)
        self._init_fc()

    def _init_fc(self):
        for m in self.fc_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        feat = self.backbone(x)
        flat = feat.view(feat.size(0), -1)
        return self.fc_head(flat)
