"""Small UNet used for both PGDN heads (reconstruction and segmentation).

Three downsampling stages + bottleneck; at ``base_channels=16`` and 128x128
inputs this is ~0.53M parameters, so a two-head PGDN is ~1.1M parameters and
trains comfortably on CPU.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, cin: int, cskip: int, cout: int):
        super().__init__()
        self.reduce = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
        self.block = ConvBlock(cout + cskip, cout)

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.reduce(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


class UNet(nn.Module):
    """Compact UNet.

    Args:
        in_channels: input channels (1 for recon head, 2/3 for seg head).
        out_channels: output channels (1).
        base_channels: feature width multiplier (default 16).
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 16):
        super().__init__()
        b = base_channels
        self.enc1 = ConvBlock(in_channels, b)      # 128
        self.enc2 = ConvBlock(b, b * 2)            # 64
        self.enc3 = ConvBlock(b * 2, b * 4)        # 32
        self.bottleneck = ConvBlock(b * 4, b * 8)  # 16
        self.dec3 = UpBlock(b * 8, b * 4, b * 4)   # 32
        self.dec2 = UpBlock(b * 4, b * 2, b * 2)   # 64
        self.dec1 = UpBlock(b * 2, b, b)           # 128
        self.head = nn.Conv2d(b, out_channels, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        z = self.bottleneck(self.pool(s3))
        d3 = self.dec3(z, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)
        return self.head(d1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
