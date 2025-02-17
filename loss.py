import torch
import torch.nn as nn

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, preds, targets):

        bce_loss = self.bce(preds, targets)

        smooth = 1e-6
        preds_flat = torch.sigmoid(preds).view(-1)
        targets_flat = targets.view(-1)
        intersection = (preds_flat * targets_flat).sum()
        dice_score = (2.0 * intersection + smooth) / (preds_flat.sum() + targets_flat.sum() + smooth)
        dice_loss = -torch.log(dice_score)

        return bce_loss + dice_loss