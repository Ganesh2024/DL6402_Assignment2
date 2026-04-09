"""U-Net style semantic segmentation built on the VGG11 encoder.

Architecture overview
---------------------
Encoder  (contracting path)  = VGG11 stages 1-5  (see vgg11.py)
Decoder  (expansive  path)   = 5 symmetric upsampling stages

Skip connections
    At each decoder stage the transposed-conv output is concatenated
    (channel-wise) with the *pre-pool* feature map from the matching encoder
    stage before the next set of convolutions.  This restores fine spatial
    detail that is lost during pooling.

Upsampling
    ConvTranspose2d with stride=2 doubles the spatial dimensions — bilinear
    interpolation and unpooling are explicitly *not* used per the spec.

Loss function choice  (stated in docstring as required)
    We use a combination of Cross-Entropy + Dice loss.
    * Cross-Entropy handles class imbalance at the pixel level and gives
      stable gradients across all classes.
    * Dice loss directly optimises the overlap metric that we are evaluated
      on, and is robust to the foreground/background imbalance present in the
      Pet trimap (most pixels are background).
    The two terms are weighted equally (0.5 each) so that neither dominates.

Output
    Raw logits [B, num_classes, H, W].  Softmax/argmax is applied externally
    during inference.
"""

import torch
import torch.nn as nn

from .vgg11 import VGG11Encoder
from .layers import CustomDropout


def _up_block(in_ch: int, out_ch: int) -> nn.ConvTranspose2d:
    """Transposed conv that doubles spatial dimensions."""
    return nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)


def _double_conv(in_ch: int, out_ch: int) -> nn.Sequential:
    """Two Conv-BN-ReLU operations applied after a skip-connection concat."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )



class VGG11UNet(nn.Module):
    """
    VGG11 encoder fused with a symmetric transposed-conv decoder.

    Encoder skip-map channels (after BN-ReLU, *before* pooling):
        s1: 64   @ 112×112
        s2: 128  @  56×56
        s3: 256  @  28×28
        s4: 512  @  14×14
        s5: 512  @   7×7   ← bottleneck (after pool)

    Decoder channel progression (mirror of encoder):
        up5: 512 → 512,  concat s4 (512)  → conv 1024 → 512
        up4: 512 → 256,  concat s3 (256)  → conv  512 → 256
        up3: 256 → 128,  concat s2 (128)  → conv  256 → 128
        up2: 128 →  64,  concat s1 ( 64)  → conv  128 →  64
        up1:  64 →  32                     → conv   32 →  32
        final 1×1 conv                     → num_classes
    """

    def __init__(
        self,
        num_classes: int = 3,
        in_channels: int = 3,
        dropout_p: float = 0.5,
    ):
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels)

        self.up5 = _up_block(512, 512)
        self.up4 = _up_block(512, 256)
        self.up3 = _up_block(256, 128)
        self.up2 = _up_block(128, 64)
        self.up1 = _up_block(64,  32)

        self.dec5 = _double_conv(1024, 512)
        self.dec4 = _double_conv(512,  256)
        self.dec3 = _double_conv(256,  128)
        self.dec2 = _double_conv(128,  64)
        self.dec1 = _double_conv(32,   32)

        self.drop = CustomDropout(p=dropout_p)

        # 1×1 conv to produce per-pixel class logits
        self.output_conv = nn.Conv2d(32, num_classes, kernel_size=1)

        self._init_decoder_weights()

    def _init_decoder_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, in_channels, 224, 224] normalised image

        Returns:
            logits [B, num_classes, 224, 224]
        """
        bottleneck, skips = self.encoder(x, return_features=True)
        # skips keys: 's1' (64@112), 's2' (128@56), 's3' (256@28),
        #             's4' (512@14), 's5' (512@7)

        # Decoder stage 5: upsample bottleneck, fuse with s4
        d5 = self.up5(bottleneck)                           
        d5 = torch.cat([d5, skips["s4"]], dim=1)            
        d5 = self.drop(self.dec5(d5))                       

        # Decoder stage 4: upsample, fuse with s3
        d4 = self.up4(d5)                                   
        d4 = torch.cat([d4, skips["s3"]], dim=1)            
        d4 = self.drop(self.dec4(d4))                       

        # Decoder stage 3: upsample, fuse with s2
        d3 = self.up3(d4)                                   
        d3 = torch.cat([d3, skips["s2"]], dim=1)            
        d3 = self.drop(self.dec3(d3))                       

        # Decoder stage 2: upsample, fuse with s1
        d2 = self.up2(d3)                                   
        d2 = torch.cat([d2, skips["s1"]], dim=1)            
        d2 = self.drop(self.dec2(d2))                       

        # Decoder stage 1: upsample to full resolution, no skip available
        d1 = self.up1(d2)                                   
        d1 = self.dec1(d1)                                  

        out = self.output_conv(d1)                          
        return out
