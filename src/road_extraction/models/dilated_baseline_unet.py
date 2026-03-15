from __future__ import annotations

import torch
from torch import nn

from .layers import ConvBlock, DownSample, OutputLayer, UpSample


class DilatedConvBranch(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DilatedContextBlock(nn.Module):
    def __init__(self, channels: int, dilations: tuple[int, ...] = (1, 2, 4, 8)) -> None:
        super().__init__()
        self.branches = nn.ModuleList([DilatedConvBranch(channels, dilation) for dilation in dilations])
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * len(dilations), channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        merged = torch.cat([branch(x) for branch in self.branches], dim=1)
        return x + self.fuse(merged)


class DilatedBaselineUNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1, init_channels: int = 32, apply_sigmoid: bool = False) -> None:
        super().__init__()
        bottleneck_channels = init_channels * 16
        self.enc1 = ConvBlock(in_channels, init_channels)
        self.enc2 = DownSample(init_channels, init_channels * 2)
        self.enc3 = DownSample(init_channels * 2, init_channels * 4)
        self.enc4 = DownSample(init_channels * 4, init_channels * 8)
        self.bottleneck = DownSample(init_channels * 8, bottleneck_channels)
        self.context = DilatedContextBlock(bottleneck_channels)
        self.dec4 = UpSample(bottleneck_channels + init_channels * 8, init_channels * 8)
        self.dec3 = UpSample(init_channels * 8 + init_channels * 4, init_channels * 4)
        self.dec2 = UpSample(init_channels * 4 + init_channels * 2, init_channels * 2)
        self.dec1 = UpSample(init_channels * 2 + init_channels, init_channels)
        self.output = OutputLayer(init_channels, out_channels, apply_sigmoid=apply_sigmoid)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        bottleneck = self.context(self.bottleneck(e4))
        d4 = self.dec4(bottleneck, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        return self.output(d1)

    def count_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
