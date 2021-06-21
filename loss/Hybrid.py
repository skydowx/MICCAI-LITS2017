"""

Dice loss + BCE loss
"""

import torch
import torch.nn as nn


class HybridLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.bce_loss = nn.BCELoss()
        self.bce_weight = 1.0

    def forward(self, pred, target):

        pred = pred.squeeze(dim=1)

        smooth = 1

        # Definition of dice coefficient
        dice = 2 * (pred * target).sum(dim=1).sum(dim=1).sum(dim=1) / (pred.pow(2).sum(dim=1).sum(dim=1).sum(dim=1) +
                                            target.pow(2).sum(dim=1).sum(dim=1).sum(dim=1) + smooth)

        # What is returned is dice distance + 　 binarized cross entropy loss
        return torch.clamp((1 - dice).mean(), 0, 1) + self.bce_loss(pred, target) * self.bce_weight
