"""
losses.py
=========
Loss functions for binary segmentation.

DiceLoss        : standard soft-Dice (good for class imbalance)
BCEDiceLoss     : 0.5 * BCE  +  0.5 * Dice  (best empirical combo)
FocalLoss       : focuses training on hard pixels
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits  : (B, C, H, W)  raw model output
        targets : (B, H, W)     long integer class labels
        """
        num_classes = logits.size(1)
        probs       = F.softmax(logits, dim=1)

        # one-hot encode targets
        targets_oh  = F.one_hot(targets, num_classes)          # B H W C
        targets_oh  = targets_oh.permute(0, 3, 1, 2).float()  # B C H W

        # only compute dice on the foreground class (class 1)
        probs_fg    = probs[:, 1]
        targets_fg  = targets_oh[:, 1]

        intersection = (probs_fg * targets_fg).sum(dim=(1, 2))
        union        = probs_fg.sum(dim=(1, 2)) + targets_fg.sum(dim=(1, 2))
        dice         = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """Combines CrossEntropy and Dice — best for segmentation with imbalance."""

    def __init__(self, bce_weight: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice       = DiceLoss(smooth)
        self.ce         = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        return (self.bce_weight * self.ce(logits, targets) +
                (1 - self.bce_weight) * self.dice(logits, targets))


class FocalLoss(nn.Module):
    """Focal loss — good when background dominates heavily."""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt      = torch.exp(-ce_loss)
        focal   = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal.mean()
