"""
model.py
========
U-Net implementation for inhibition zone segmentation.

Architecture:
  - Encoder: 4 down-sampling blocks (Conv→BN→ReLU ×2, MaxPool)
  - Bottleneck
  - Decoder: 4 up-sampling blocks with skip connections
  - Output: 1×1 conv → 2-class softmax (background / zone)

Optionally uses a ResNet-34 encoder (transfer learning) when
use_pretrained=True, which significantly helps with small datasets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── building blocks ───────────────────────────────────────────────────────────

def _make_norm(norm_type: str, num_channels: int) -> nn.Module:
    """
    'batch' : nn.BatchNorm2d — original default. Needs a reasonably large
              batch size for stable running statistics; with batch_size=4
              at img_size=1024 this is a likely source of the noisy,
              spiky validation curves observed during training.
    'group' : nn.GroupNorm — computed per-sample, independent of batch
              size, so it doesn't have that instability. Standard choice
              for small-batch, large-image segmentation (e.g. nnU-Net
              uses InstanceNorm for the same reason). Recommended for
              img_size=1024 / batch_size=4 training.
    """
    if norm_type == "group":
        num_groups = min(8, num_channels)  # 8 groups, or fewer if channels < 8
        return nn.GroupNorm(num_groups, num_channels)
    return nn.BatchNorm2d(num_channels)


class DoubleConv(nn.Module):
    """Two consecutive Conv2d → Norm → ReLU blocks."""

    def __init__(self, in_ch: int, out_ch: int, mid_ch: int = None,
                 norm_type: str = "batch"):
        super().__init__()
        mid_ch = mid_ch or out_ch
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,  mid_ch, 3, padding=1, bias=False),
            _make_norm(norm_type, mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, 3, padding=1, bias=False),
            _make_norm(norm_type, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """MaxPool → DoubleConv (encoder step)."""

    def __init__(self, in_ch: int, out_ch: int, norm_type: str = "batch"):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch, norm_type=norm_type),
        )

    def forward(self, x):
        return self.pool_conv(x)


class Up(nn.Module):
    """Bilinear up-sample → concat skip → DoubleConv (decoder step)."""

    def __init__(self, in_ch: int, out_ch: int, norm_type: str = "batch"):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear",
                                align_corners=True)
        self.conv = DoubleConv(in_ch, out_ch, in_ch // 2, norm_type=norm_type)

    def forward(self, x, skip):
        x = self.up(x)
        # pad if sizes don't match exactly
        dh = skip.size(2) - x.size(2)
        dw = skip.size(3) - x.size(3)
        x  = F.pad(x, [dw//2, dw - dw//2, dh//2, dh - dh//2])
        x  = torch.cat([skip, x], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_ch: int, num_classes: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, num_classes, 1)

    def forward(self, x):
        return self.conv(x)


# ── vanilla U-Net ─────────────────────────────────────────────────────────────

class UNet(nn.Module):
    """
    Standard U-Net.

    Parameters
    ----------
    in_channels  : 3 (RGB)
    num_classes  : 2 (background, inhibition-zone)
    base_filters : feature-map depth at first level (default 64)
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 2,
                 base_filters: int = 64, norm_type: str = "batch"):
        super().__init__()
        f = base_filters

        self.inc   = DoubleConv(in_channels, f, norm_type=norm_type)
        self.down1 = Down(f,    f*2,  norm_type=norm_type)
        self.down2 = Down(f*2,  f*4,  norm_type=norm_type)
        self.down3 = Down(f*4,  f*8,  norm_type=norm_type)
        self.down4 = Down(f*8,  f*16, norm_type=norm_type)

        self.up1   = Up(f*16 + f*8,  f*8, norm_type=norm_type)
        self.up2   = Up(f*8  + f*4,  f*4, norm_type=norm_type)
        self.up3   = Up(f*4  + f*2,  f*2, norm_type=norm_type)
        self.up4   = Up(f*2  + f,    f,   norm_type=norm_type)
        self.outc  = OutConv(f, num_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x  = self.up1(x5, x4)
        x  = self.up2(x,  x3)
        x  = self.up3(x,  x2)
        x  = self.up4(x,  x1)
        return self.outc(x)             # logits: (B, num_classes, H, W)


# ── lightweight U-Net for CPU / low-VRAM ─────────────────────────────────────

class UNetSmall(UNet):
    """Same architecture with base_filters=32 for CPU training."""
    def __init__(self, in_channels=3, num_classes=2, norm_type: str = "batch"):
        super().__init__(in_channels, num_classes, base_filters=32, norm_type=norm_type)


# ── factory ───────────────────────────────────────────────────────────────────

def build_model(size: str = "full", num_classes: int = 2,
                device: str = "cpu", norm_type: str = "batch") -> nn.Module:
    """
    size      : 'full'  → UNet(base_filters=64)   ~31M params
                'small' → UNet(base_filters=32)   ~8M  params  (recommended for CPU)
    norm_type : 'batch' → nn.BatchNorm2d (original default; backward-compatible
                           with checkpoints trained before this option existed)
                'group' → nn.GroupNorm (recommended for small batch_size + large
                           img_size, e.g. batch_size=4 @ img_size=1024 — see
                           _make_norm() docstring)

    IMPORTANT: norm_type changes the model's parameter names/shapes.
    A checkpoint trained with norm_type='batch' will NOT load into a
    norm_type='group' model and vice versa — always pass the SAME
    --norm value at train time and at inference/eval time for a given
    checkpoint.
    """
    if size == "small":
        model = UNetSmall(num_classes=num_classes, norm_type=norm_type)
    else:
        model = UNet(num_classes=num_classes, norm_type=norm_type)

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: UNet-{size}  |  Norm: {norm_type}  |  "
          f"Parameters: {n_params:,}  |  Device: {device}")
    return model