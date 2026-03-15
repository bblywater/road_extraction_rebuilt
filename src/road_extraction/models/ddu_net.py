from __future__ import annotations

import warnings

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet50_Weights, resnet50


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x))
        return torch.sigmoid(scale)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        max_ = torch.amax(x, dim=1, keepdim=True)
        return torch.sigmoid(self.conv(torch.cat([avg, max_], dim=1)))


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7) -> None:
        super().__init__()
        self.channel = ChannelAttention(channels, reduction=reduction)
        self.spatial = SpatialAttention(kernel_size=spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel(x)
        x = x * self.spatial(x)
        return x


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dilation: int = 1) -> None:
        super().__init__()
        padding = dilation if kernel_size == 3 else 0
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.fuse = nn.Sequential(
            ConvNormAct(in_channels + skip_channels, out_channels, kernel_size=3),
            ConvNormAct(out_channels, out_channels, kernel_size=3),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.fuse(x)


class DCAM(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, reduction: int = 16) -> None:
        super().__init__()
        branch_channels = out_channels // 5
        self.branch_1x1 = ConvNormAct(in_channels, branch_channels, kernel_size=1)
        self.branch_d2 = ConvNormAct(in_channels, branch_channels, kernel_size=3, dilation=2)
        self.branch_d4 = ConvNormAct(in_channels, branch_channels, kernel_size=3, dilation=4)
        self.branch_gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
        )
        self.branch_cbam = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            CBAM(branch_channels, reduction=reduction),
        )
        self.project = nn.Sequential(
            nn.Conv2d(branch_channels * 5, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial_size = x.shape[-2:]
        gap = self.branch_gap(x)
        gap = F.interpolate(gap, size=spatial_size, mode="bilinear", align_corners=False)
        merged = torch.cat(
            [
                self.branch_1x1(x),
                self.branch_d2(x),
                self.branch_d4(x),
                gap,
                self.branch_cbam(x),
            ],
            dim=1,
        )
        return self.project(merged)


class DDUHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct(in_channels, in_channels, kernel_size=3),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        logits = self.block(x)
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)


class DDUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        init_channels: int = 32,
        pretrained: bool = True,
        aux_weight: float = 0.4,
        dcam_channels: int = 512,
        decoder_channels: tuple[int, int, int, int] = (512, 256, 128, 64),
        cbam_reduction: int = 16,
        freeze_encoder_bn: bool = True,
    ) -> None:
        super().__init__()
        if in_channels != 3:
            raise ValueError("DDUNet requires in_channels=3 because the encoder is ResNet-50 based.")
        self.aux_weight = aux_weight
        self.freeze_encoder_bn = freeze_encoder_bn
        encoder = self._build_encoder(pretrained=pretrained)
        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.maxpool = encoder.maxpool
        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

        self.dcam = DCAM(2048, dcam_channels, reduction=cbam_reduction)
        dec4, dec3, dec2, dec1 = decoder_channels
        self.main_dec4 = DecoderBlock(dcam_channels, 1024, dec4)
        self.main_dec3 = DecoderBlock(dec4, 512, dec3)
        self.main_dec2 = DecoderBlock(dec3, 256, dec2)
        self.main_dec1 = DecoderBlock(dec2, 64, dec1)
        self.main_head = DDUHead(dec1, out_channels)

        self.aux_dec4 = DecoderBlock(dcam_channels, 1024, 256)
        self.aux_dec3 = DecoderBlock(256, 512, 128)
        self.aux_head = nn.Sequential(
            ConvNormAct(128, 128, kernel_size=3),
            nn.Conv2d(128, out_channels, kernel_size=1),
        )

    def _build_encoder(self, pretrained: bool) -> nn.Module:
        weights = None
        if pretrained:
            try:
                weights = ResNet50_Weights.IMAGENET1K_V2
            except Exception:
                weights = None
        try:
            return resnet50(weights=weights)
        except Exception as exc:
            warnings.warn(f"Falling back to randomly initialized ResNet-50 encoder: {exc}", RuntimeWarning)
            return resnet50(weights=None)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        output_size = x.shape[-2:]
        c1 = self.stem(x)
        c2 = self.layer1(self.maxpool(c1))
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        context = self.dcam(c5)

        main = self.main_dec4(context, c4)
        main = self.main_dec3(main, c3)
        main = self.main_dec2(main, c2)
        main = self.main_dec1(main, c1)
        logits = self.main_head(main, output_size)

        aux = self.aux_dec4(context, c4)
        aux = self.aux_dec3(aux, c3)
        aux_logits = self.aux_head(aux)
        return {"logits": logits, "aux_logits": aux_logits}

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
