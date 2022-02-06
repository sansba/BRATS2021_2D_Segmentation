import torch
import torch.nn as nn

import functional


#DICE LOSS
class DiceLoss(nn.Module):
    def __init__(self):
        """Computes the Sørensen–Dice loss.\n
        Formula: 1 - (2 * (A ∩ B)) / (A + B) \n
        A and B predicted and ground truth values.
        """
        super(DiceLoss, self).__init__()
        self.one_hot_encoder = functional.OneHotEncoder()

    def forward(self, prediction, ground_truth, smooth=1e-9):
        """Args:
            - prediction: 4 dimensional tensor (B, C, H, W)
            - ground_truth: 3 dimensional tensor (B, H, W) \n 
              where B: batch size,   C: number of classes,   H: height,   W: width
            - smooth: for numerical stability
            """
        n_classes = prediction.shape[1]

        softmax_prediction = torch.softmax(prediction, dim=1)
        binary_ground_truth = self.one_hot_encoder(ground_truth, n_classes)

        intersections = torch.sum(softmax_prediction * binary_ground_truth, dim=(2, 3))
        unions = torch.sum(softmax_prediction + binary_ground_truth, dim=(2, 3))

        dice_loss = (2 * intersections + smooth) / (unions + smooth)

        return 1 - torch.mean(dice_loss)



#JACCARD LOSS
class JaccardLoss(nn.Module):
    def __init__(self):
        """Computes Jaccard Loss (IOU Loss). \n
        Formula: 1 - (A ∩ B) / (A U B) \n
        A and B predicted and ground truth values.
        """
        super(JaccardLoss, self).__init__()
        self.one_hot_encoder = functional.OneHotEncoder()

    def forward(self, prediction, ground_truth, smooth=1e-9):
        """Args:
            - prediction: 4 dimensional tensor (B, C, H, W)
            - ground_truth: 3 dimensional tensor (B, H, W) \n 
              where B: batch size,   C: number of classes,   H: height,   W: width
            - smooth: for numerical stability
            """
        n_classes = prediction.shape[1]

        softmax_prediction = torch.softmax(prediction, dim=1)
        binary_ground_truth = self.one_hot_encoder(ground_truth, n_classes)

        intersections = torch.sum(softmax_prediction * binary_ground_truth, dim=(2, 3))
        unions = torch.sum(softmax_prediction + binary_ground_truth, dim=(2, 3)) - intersections

        jaccard = (intersections + smooth) / (unions + smooth)

        return 1 - torch.mean(jaccard)




#TVERSKY LOSS
class TverskyLoss(nn.Module):
    def __init__(self):
        """Computes Tversky Loss. \n
        Formula: 
            TP / (TP + FN * alpha + FP * beta) \n
            - TP: True Positive
            - FN: False Negative
            - FP: False Positive
            - alpha: coefficient for FN
            - beta: coefficient for FP

        """
        super(TverskyLoss, self).__init__()
        self.one_hot_encoder = functional.OneHotEncoder()

    def forward(self, prediction, ground_truth, alpha, beta, smooth=1e-9):
        """Args:
            - prediction: 4 dimensional tensor (B, C, H, W)
            - ground_truth: 3 dimensional tensor (B, H, W) \n
              where B: batch size,   C: number of classes,   H: height,   W: width
            - alpha: coefficient of FN
            - beta: coefficient of FP
            - smooth: for numerical stability
            """
        n_classes = prediction.shape[1]

        softmax_prediction = torch.softmax(prediction, dim=1)
        binary_ground_truth = self.one_hot_encoder(ground_truth, n_classes)

        TP = torch.sum(softmax_prediction * binary_ground_truth, dim=(2, 3))
        FP = torch.sum(softmax_prediction * (1 - binary_ground_truth), dim=(2, 3))
        FN = torch.sum((1 - softmax_prediction) * binary_ground_truth, dim=(2, 3))

        tversky = (TP + smooth) / (TP + alpha * FN + beta * FP)

        return 1 - torch.mean(tversky)




#FOCAL LOSS
class FocalLoss(nn.Module):
    def __init__(self):
        """Computes Focal Loss. \n
        Formula: - (1 - p) ^ γ * log(p) \n
        - p: the probability of the corresponding classes.
        - γ: exponential coefficinet.
        """
        super(FocalLoss, self).__init__()
        self.one_hot_encoder = functional.OneHotEncoder()

    def forward(self, prediction, ground_truth, gamma=5):
        """Args:
            - prediction: 4 dimensional tensor (B, C, H, W)
            - ground_truth: 3 dimensional tensor (B, H, W) \n 
              where B: batch size,   C: number of classes,   H: height,   W: width
            - gamma: coefficient of loss function
            """
        n_classes = prediction.shape[1]

        softmax_targets = torch.softmax(prediction, dim=1)
        binary_ground_truth = self.one_hot_encoder(ground_truth, n_classes)

        fl_loss = -binary_ground_truth * torch.pow((1 - softmax_targets), gamma) * torch.log(softmax_targets)
        
        return fl_loss.mean()



#CROSS ENTROPY LOSS
class CrossEntropyLoss(nn.Module):
    def __init__(self):
        """Calculates Cross Entropy Loss. \n
        Formula: -log(p) \n
         p is the probability of the corresponding classes.
        """
        super(CrossEntropyLoss, self).__init__()
        self.one_hot_encoder = functional.OneHotEncoder()


    def forward(self, prediction, ground_truth):
        """Args:
            - prediction: 4 dimensional tensor (B, C, H, W)
            - ground_truth: 3 dimensional tensor (B, H, W) \n 
              where B: batch size,   C: number of classes,   H: height,   W: width
            """
        n_classes = prediction.shape[1]

        softmax_prediction = torch.softmax(prediction, dim=1)
        binary_ground_truth = self.one_hot_encoder(ground_truth, n_classes)    

        ce_loss = -binary_ground_truth * torch.log(softmax_prediction)
        
        return ce_loss.mean()



