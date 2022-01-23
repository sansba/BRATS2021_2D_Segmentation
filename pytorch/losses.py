import torch.nn as nn


#Dice Loss
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, targets, inputs, smooth=1e-6):
        total_dice = 0

        for i in range(targets.shape[1]):
            class_0 = (inputs == i).int()
            target_0 = targets[:, i, :, :].int()

            intersect = (class_0 & target_0).sum()
            union = class_0.sum() + target_0.sum()

            dice = (2. * intersect + smooth) / (union + smooth)
            total_dice += dice

        return 1 - total_dice / targets.shape[1]




#Mean IOU
class MeanIOU(nn.Module):
    def __init__(self):
        """Calculates mean intersection over union accuracy"""
        super(MeanIOU, self).__init__()

    
    def forward(self, targets, inputs, smooth=1e-6):
        total_iou = 0

        for i in range(targets.shape[1]):
            class_0 = (inputs == i).int()
            target_0 = targets[:, i, :, :].int()

            intersect = (class_0 & target_0).sum()
            union = class_0.sum() + target_0.sum() - intersect

            iou = (intersect + smooth) / (union + smooth)
            total_iou += iou

        return total_iou / targets.shape[1]