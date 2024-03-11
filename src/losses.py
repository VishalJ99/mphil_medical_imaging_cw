import torch
import torch.nn as nn


class SoftDiceLoss(nn.Module):
    """
    Soft dice loss for binary segmentation. Designed to work with the
    raw logits output from the model. Assumes input is a tensors are of shape
    (N, 1, H, W) where N is the batch size, 1 is the number of classes, H and
    W are the height and width of the mask.

    Outputs dice loss for each image in the batch.

    NOTE:
    For multi-class segmentation, the softmax activation function should
    be applied instead.
    """

    def __init__(self, eps=1e-4, apply_sigmoid=True):
        # apply sigmoid added for debugging purposes.
        super(SoftDiceLoss, self).__init__()
        self.eps = eps
        self.apply_sigmoid = apply_sigmoid

    def forward(self, preds, targets, apply_sigmoid=True):
        if self.apply_sigmoid:
            preds = torch.sigmoid(preds)

        intersection = torch.sum(preds * targets, dim=(2, 3))
        denominator = torch.sum(preds, dim=(2, 3)) + torch.sum(targets, dim=(2, 3))

        dice = (2 * intersection) / (denominator + self.eps)
        dice_loss = 1 - dice
        return dice_loss


class SoftDiceBCECombinedLoss(nn.Module):
    def __init__(self, dice_weight=0.5, eps=1):
        super(SoftDiceBCECombinedLoss, self).__init__()
        self.dice_weight = dice_weight
        self.bce_weight = 1 - dice_weight
        self.dice_loss = SoftDiceLoss(eps=eps)
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, preds, targets):
        dice_loss = self.dice_loss(preds, targets)
        bce_loss = self.bce_loss(preds, targets)
        combined_loss = self.dice_weight * dice_loss + self.bce_weight * bce_loss
        return combined_loss
