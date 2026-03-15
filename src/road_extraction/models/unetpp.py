from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .layers import ConvBlock, OutputLayer


class UNetPPUp(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.project = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != target.shape[-2:]:
            diff_y = target.size(2) - x.size(2)
            diff_x = target.size(3) - x.size(3)
            x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        return self.project(x)


class UNetPP(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        init_channels: int = 64,
        apply_sigmoid: bool = False,
    ) -> None:
        super().__init__()
        filters = [init_channels, init_channels * 2, init_channels * 4, init_channels * 8, init_channels * 16]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv0_0 = ConvBlock(in_channels, filters[0])
        self.conv1_0 = ConvBlock(filters[0], filters[1])
        self.conv2_0 = ConvBlock(filters[1], filters[2])
        self.conv3_0 = ConvBlock(filters[2], filters[3])
        self.conv4_0 = ConvBlock(filters[3], filters[4])

        self.up1 = UNetPPUp(filters[1], filters[0])
        self.up2 = UNetPPUp(filters[2], filters[1])
        self.up3 = UNetPPUp(filters[3], filters[2])
        self.up4 = UNetPPUp(filters[4], filters[3])

        self.conv0_1 = ConvBlock(filters[0] * 2, filters[0])
        self.conv1_1 = ConvBlock(filters[1] * 2, filters[1])
        self.conv2_1 = ConvBlock(filters[2] * 2, filters[2])
        self.conv3_1 = ConvBlock(filters[3] * 2, filters[3])

        self.conv0_2 = ConvBlock(filters[0] * 3, filters[0])
        self.conv1_2 = ConvBlock(filters[1] * 3, filters[1])
        self.conv2_2 = ConvBlock(filters[2] * 3, filters[2])

        self.conv0_3 = ConvBlock(filters[0] * 4, filters[0])
        self.conv1_3 = ConvBlock(filters[1] * 4, filters[1])

        self.conv0_4 = ConvBlock(filters[0] * 5, filters[0])

        self.out = OutputLayer(filters[0], out_channels, apply_sigmoid=apply_sigmoid)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up1(x1_0, x0_0)], dim=1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up2(x2_0, x1_0)], dim=1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up1(x1_1, x0_0)], dim=1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up3(x3_0, x2_0)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up2(x2_1, x1_0)], dim=1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up1(x1_2, x0_0)], dim=1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up4(x4_0, x3_0)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up3(x3_1, x2_0)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up2(x2_2, x1_0)], dim=1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up1(x1_3, x0_0)], dim=1))

        return self.out(x0_4)

    def count_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
