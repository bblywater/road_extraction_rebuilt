from __future__ import annotations

import torch
from torch import nn

from .layers import ConvBlock, DownSample, OutputLayer, UpSample


class ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(1, in_channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(nn.Linear(in_channels, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, in_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, _, _ = x.size()
        avg = self.mlp(self.avg_pool(x).view(batch, channels)).view(batch, channels, 1, 1)
        max_ = self.mlp(self.max_pool(x).view(batch, channels)).view(batch, channels, 1, 1)
        return torch.sigmoid(avg + max_)


class SpatialAttention(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int = 5) -> None:
        super().__init__()
        hidden = max(1, in_channels // 2)
        self.conv_h = nn.Conv2d(in_channels, hidden, kernel_size=(1, kernel_size), padding=(0, kernel_size // 2))
        self.conv_v = nn.Conv2d(in_channels, hidden, kernel_size=(kernel_size, 1), padding=(kernel_size // 2, 0))
        self.fuse = nn.Conv2d(hidden, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.fuse(self.conv_h(x) + self.conv_v(x)))


class DLAM(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16, spatial_kernel: int = 5) -> None:
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction=reduction)
        self.spatial_attention = SpatialAttention(in_channels, kernel_size=spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channel_weight = self.channel_attention(x)
        spatial_weight = self.spatial_attention(x)
        combined = channel_weight * spatial_weight
        return x + x * combined


class DLGUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        init_channels: int = 32,
        dlam_reduction: int = 16,
        dlam_spatial_kernel: int = 5,
    ) -> None:
        super().__init__()
        self.enc1 = ConvBlock(in_channels, init_channels)
        self.enc2 = DownSample(init_channels, init_channels * 2)
        self.enc3 = DownSample(init_channels * 2, init_channels * 4)
        self.enc4 = DownSample(init_channels * 4, init_channels * 8)
        self.bottleneck = DownSample(init_channels * 8, init_channels * 16)
        self.dec4 = UpSample(init_channels * 16 + init_channels * 8, init_channels * 8)
        self.dec3 = UpSample(init_channels * 8 + init_channels * 4, init_channels * 4)
        self.dec2 = UpSample(init_channels * 4 + init_channels * 2, init_channels * 2)
        self.dec1 = UpSample(init_channels * 2 + init_channels, init_channels)
        self.out = OutputLayer(init_channels, out_channels, apply_sigmoid=False)
        self.dlam1 = DLAM(init_channels, reduction=dlam_reduction, spatial_kernel=dlam_spatial_kernel)
        self.dlam2 = DLAM(init_channels * 2, reduction=dlam_reduction, spatial_kernel=dlam_spatial_kernel)
        self.dlam3 = DLAM(init_channels * 4, reduction=dlam_reduction, spatial_kernel=dlam_spatial_kernel)
        self.dlam4 = DLAM(init_channels * 8, reduction=dlam_reduction, spatial_kernel=dlam_spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        bottleneck = self.bottleneck(enc4)
        dec4 = self.dec4(bottleneck, self.dlam4(enc4))
        dec3 = self.dec3(dec4, self.dlam3(enc3))
        dec2 = self.dec2(dec3, self.dlam2(enc2))
        dec1 = self.dec1(dec2, self.dlam1(enc1))
        return self.out(dec1)

    def count_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
