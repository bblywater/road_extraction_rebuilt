from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .layers import ConvBlock, DownSample, OutputLayer, UpSample


class AttentionGate(nn.Module):
    def __init__(self, skip_channels: int, gate_channels: int, inter_channels: int) -> None:
        super().__init__()
        self.skip_proj = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.gate_proj = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.attention = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, skip: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        if gate.shape[-2:] != skip.shape[-2:]:
            gate = F.interpolate(gate, size=skip.shape[-2:], mode="bilinear", align_corners=True)
        alpha = self.attention(self.skip_proj(skip) + self.gate_proj(gate))
        return skip * alpha


class AttentionUNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1, init_channels: int = 32, apply_sigmoid: bool = False) -> None:
        super().__init__()
        self.enc1 = ConvBlock(in_channels, init_channels)
        self.enc2 = DownSample(init_channels, init_channels * 2)
        self.enc3 = DownSample(init_channels * 2, init_channels * 4)
        self.enc4 = DownSample(init_channels * 4, init_channels * 8)
        self.bottleneck = DownSample(init_channels * 8, init_channels * 16)

        self.att4 = AttentionGate(init_channels * 8, init_channels * 16, init_channels * 4)
        self.att3 = AttentionGate(init_channels * 4, init_channels * 8, init_channels * 2)
        self.att2 = AttentionGate(init_channels * 2, init_channels * 4, init_channels)
        self.att1 = AttentionGate(init_channels, init_channels * 2, max(1, init_channels // 2))

        self.dec4 = UpSample(init_channels * 16 + init_channels * 8, init_channels * 8)
        self.dec3 = UpSample(init_channels * 8 + init_channels * 4, init_channels * 4)
        self.dec2 = UpSample(init_channels * 4 + init_channels * 2, init_channels * 2)
        self.dec1 = UpSample(init_channels * 2 + init_channels, init_channels)
        self.output = OutputLayer(init_channels, out_channels, apply_sigmoid=apply_sigmoid)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        bottleneck = self.bottleneck(e4)

        d4 = self.dec4(bottleneck, self.att4(e4, bottleneck))
        d3 = self.dec3(d4, self.att3(e3, d4))
        d2 = self.dec2(d3, self.att2(e2, d3))
        d1 = self.dec1(d2, self.att1(e1, d2))
        return self.output(d1)

    def count_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
