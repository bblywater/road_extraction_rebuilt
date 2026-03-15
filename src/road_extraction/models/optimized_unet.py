from __future__ import annotations

import torch
from torch import nn


class OptimizedUNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1, init_channels: int = 32) -> None:
        super().__init__()
        self.encoder1 = self._conv_block(in_channels, init_channels)
        self.encoder2 = self._conv_block(init_channels, init_channels * 2)
        self.encoder3 = self._conv_block(init_channels * 2, init_channels * 4)
        self.encoder4 = self._conv_block(init_channels * 4, init_channels * 8)
        self.bottleneck = self._conv_block(init_channels * 8, init_channels * 16)
        self.upconv4 = nn.ConvTranspose2d(init_channels * 16, init_channels * 8, kernel_size=2, stride=2)
        self.decoder4 = self._conv_block(init_channels * 16, init_channels * 8)
        self.upconv3 = nn.ConvTranspose2d(init_channels * 8, init_channels * 4, kernel_size=2, stride=2)
        self.decoder3 = self._conv_block(init_channels * 8, init_channels * 4)
        self.upconv2 = nn.ConvTranspose2d(init_channels * 4, init_channels * 2, kernel_size=2, stride=2)
        self.decoder2 = self._conv_block(init_channels * 4, init_channels * 2)
        self.upconv1 = nn.ConvTranspose2d(init_channels * 2, init_channels, kernel_size=2, stride=2)
        self.decoder1 = self._conv_block(init_channels * 2, init_channels)
        self.conv_out = nn.Conv2d(init_channels, out_channels, kernel_size=1)
        self.maxpool = nn.MaxPool2d(2)

    @staticmethod
    def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.maxpool(enc1))
        enc3 = self.encoder3(self.maxpool(enc2))
        enc4 = self.encoder4(self.maxpool(enc3))
        bottleneck = self.bottleneck(self.maxpool(enc4))
        dec4 = self.decoder4(torch.cat([self.upconv4(bottleneck), enc4], dim=1))
        dec3 = self.decoder3(torch.cat([self.upconv3(dec4), enc3], dim=1))
        dec2 = self.decoder2(torch.cat([self.upconv2(dec3), enc2], dim=1))
        dec1 = self.decoder1(torch.cat([self.upconv1(dec2), enc1], dim=1))
        return self.conv_out(dec1)

    def count_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
