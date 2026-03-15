from __future__ import annotations

import warnings

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet34_Weights, resnet34

from .layers import ConvBlock, OutputLayer


class ResNetUNetDecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        if x.shape[-2:] != skip.shape[-2:]:
            diff_y = skip.size(2) - x.size(2)
            diff_x = skip.size(3) - x.size(3)
            x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        return self.conv(torch.cat([skip, x], dim=1))


class ResNet34UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        init_channels: int = 32,
        apply_sigmoid: bool = False,
        pretrained: bool = True,
        freeze_encoder_bn: bool = True,
    ) -> None:
        super().__init__()
        if in_channels != 3:
            raise ValueError("ResNet34UNet requires in_channels=3 because the encoder is ResNet34 based.")
        self.freeze_encoder_bn = freeze_encoder_bn

        encoder = self._build_encoder(pretrained=pretrained)
        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.maxpool = encoder.maxpool
        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

        self.center = ConvBlock(512, init_channels * 16)
        self.dec4 = ResNetUNetDecoderBlock(init_channels * 16, 256, init_channels * 8)
        self.dec3 = ResNetUNetDecoderBlock(init_channels * 8, 128, init_channels * 4)
        self.dec2 = ResNetUNetDecoderBlock(init_channels * 4, 64, init_channels * 2)
        self.dec1 = ResNetUNetDecoderBlock(init_channels * 2, 64, init_channels)
        self.final_refine = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            ConvBlock(init_channels, init_channels),
        )
        self.output = OutputLayer(init_channels, out_channels, apply_sigmoid=apply_sigmoid)

    def _build_encoder(self, pretrained: bool) -> nn.Module:
        weights = None
        if pretrained:
            try:
                weights = ResNet34_Weights.IMAGENET1K_V1
            except Exception:
                weights = None
        try:
            return resnet34(weights=weights)
        except Exception as exc:
            warnings.warn(f"Falling back to randomly initialized ResNet34 encoder: {exc}", RuntimeWarning)
            return resnet34(weights=None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e0 = self.stem(x)
        e1 = self.layer1(self.maxpool(e0))
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        e4 = self.layer4(e3)

        center = self.center(e4)
        d4 = self.dec4(center, e3)
        d3 = self.dec3(d4, e2)
        d2 = self.dec2(d3, e1)
        d1 = self.dec1(d2, e0)
        out = self.final_refine(d1)
        return self.output(out)

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.freeze_encoder_bn:
            for module in [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]:
                for child in module.modules():
                    if isinstance(child, nn.BatchNorm2d):
                        child.eval()
        return self

    def count_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
