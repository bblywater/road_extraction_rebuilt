from __future__ import annotations

from .attention_unet import AttentionUNet
from .baseline_unet import BaselineUNet
from .dilated_baseline_unet import DilatedBaselineUNet
from .ddu_net import DDUNet
from .dlgu_net import DLGUNet
from .optimized_unet import OptimizedUNet
from .resnet34_unet import ResNet34UNet
from .residual_vanilla_unet import ResidualVanillaUNet
from .unetpp import UNetPP
from .vanilla_unet import VanillaUNet


def build_model(config: dict):
    model_cfg = config["model"]
    name = model_cfg["name"].lower()
    common = {
        "in_channels": model_cfg.get("in_channels", 3),
        "out_channels": model_cfg.get("out_channels", 1),
        "init_channels": model_cfg.get("init_channels", 32),
    }
    if name == "baseline_unet":
        return BaselineUNet(apply_sigmoid=model_cfg.get("apply_sigmoid", False), **common)
    if name == "dilated_baseline_unet":
        return DilatedBaselineUNet(apply_sigmoid=model_cfg.get("apply_sigmoid", False), **common)
    if name == "vanilla_unet":
        return VanillaUNet(apply_sigmoid=model_cfg.get("apply_sigmoid", False), **common)
    if name == "residual_vanilla_unet":
        return ResidualVanillaUNet(apply_sigmoid=model_cfg.get("apply_sigmoid", False), **common)
    if name == "unetpp":
        return UNetPP(apply_sigmoid=model_cfg.get("apply_sigmoid", False), **common)
    if name == "resnet34_unet":
        return ResNet34UNet(
            apply_sigmoid=model_cfg.get("apply_sigmoid", False),
            pretrained=model_cfg.get("pretrained", True),
            freeze_encoder_bn=model_cfg.get("freeze_encoder_bn", True),
            **common,
        )
    if name == "attention_unet":
        return AttentionUNet(apply_sigmoid=model_cfg.get("apply_sigmoid", False), **common)
    if name == "optimized_unet":
        return OptimizedUNet(**common)
    if name == "dlgu_net":
        return DLGUNet(
            dlam_reduction=model_cfg.get("dlam", {}).get("reduction", 16),
            dlam_spatial_kernel=model_cfg.get("dlam", {}).get("spatial_kernel", 5),
            **common,
        )
    if name == "ddu_net":
        decoder_channels = model_cfg.get("decoder_channels", [512, 256, 128, 64])
        return DDUNet(
            pretrained=model_cfg.get("pretrained", True),
            aux_weight=model_cfg.get("aux_weight", 0.4),
            dcam_channels=model_cfg.get("dcam_channels", 512),
            decoder_channels=tuple(decoder_channels),
            cbam_reduction=model_cfg.get("cbam_reduction", 16),
            freeze_encoder_bn=model_cfg.get("freeze_encoder_bn", True),
            **common,
        )
    raise ValueError(f"Unsupported model: {name}")


__all__ = [
    "BaselineUNet",
    "DilatedBaselineUNet",
    "VanillaUNet",
    "ResidualVanillaUNet",
    "UNetPP",
    "ResNet34UNet",
    "AttentionUNet",
    "OptimizedUNet",
    "DLGUNet",
    "DDUNet",
    "build_model",
]
