import torch
import torch.nn as nn

import functional


#MEAN IOU SCORE
class MeanIOUScore(nn.Module):
    def __init__(self):
        """Computes IOU Score. \n
            Formula: (A ∩ B) / (A U B) \n
            A and B is predicted and ground truth values.
        """
        super(MeanIOUScore, self).__init__()
        self.one_hot_encoder = functional.OneHotEncoder()

    def forward(self, prediction, ground_truth, smooth=1e-9):
        """Args:
            - prediction: 4 dimensional tensor (B, C, H, W)
            - ground_truth: 3 dimensional tensor (B, H, W) \n 
              where B: batch size,   C: number of classes,   H: height,   W: width
            - smooth: for numerical stability
            """
        classes = prediction.shape[1]
        binary_prediction = self.one_hot_encoder(torch.argmax(prediction, dim=1), classes)
        binary_ground_truth = self.one_hot_encoder(ground_truth, classes)

        intersections = torch.sum(binary_ground_truth * binary_prediction, dim=(2, 3))
        unions = torch.sum(binary_ground_truth, dim=(2, 3)) + torch.sum(binary_prediction, dim=(2, 3)) - intersections
        iou = (intersections + smooth) / (unions + smooth)

        return torch.mean(iou).item()




#PRECISION SCORE
class PrecisionScore(nn.Module):
    def __init__(self):
        """Computes Precision Score. \n
           Formula: TP / (TP + FP) \n
         - TP: True Positives 
         - FP: False Positives
        """
        super(PrecisionScore, self).__init__()

    def forward(self, prediction, ground_truth, smooth=1e-9):
        """Args:
            - prediction: 4 dimensional tensor (B, C, H, W)
            - ground_truth: 3 dimensional tensor (B, H, W) \n 
              where B: batch size,   C: number of classes,   H: height,   W: width
            - smooth: for numerical stability
            """
        classes = prediction.shape[1]
        binary_prediction = self.one_hot_encoder(torch.argmax(prediction, dim=1), classes)
        binary_ground_truth = self.one_hot_encoder(ground_truth, classes)


        TP = torch.sum(binary_ground_truth * binary_prediction, dim=(2, 3))
        FP = torch.sum(binary_prediction, dim=(2, 3)) - TP

        precision_score = (TP + smooth) / (TP + FP + smooth)

        return torch.mean(precision_score).item()




#RECALL SCORE
class RecallScore(nn.Module):
    def __init__(self):
        """Computes Recall Score. \n
           Formula: TP / (TP + FN) \n
         - TP: True Positives 
         - FN: False Negatives
        """
        super(RecallScore, self).__init__()

    def forward(self, prediction, ground_truth, smooth=1e-9):
        """Args:
            - prediction: 4 dimensional tensor (B, C, H, W)
            - ground_truth: 3 dimensional tensor (B, H, W) \n 
              where B: batch size,   C: number of classes,   H: height,   W: width
            - smooth: for numerical stability
            """
        classes = prediction.shape[1]
        binary_prediction = self.one_hot_encoder(torch.argmax(prediction, dim=1), classes)
        binary_ground_truth = self.one_hot_encoder(ground_truth, classes)

        
        TP = torch.sum(binary_ground_truth * binary_prediction, dim=(2, 3))
        FN = torch.sum(binary_ground_truth, dim=(2, 3)) - TP

        recall_score = (TP + smooth) / (TP + FN + smooth)

        return torch.mean(recall_score).item()



#F-SCORE
class FScore(nn.Module):
    def __init__(self):
        """Computes F Score. \n
           Formula: (2 * Recall * Precision) / (Recall + Precision) \n
        """
        super(FScore, self).__init__()
        self.recall_score = RecallScore()
        self.precision_score = PrecisionScore()

    def forward(self, prediction, ground_truth, smooth=1e-9):
        """Args:
            - prediction: 4 dimensional tensor (B, C, H, W)
            - ground_truth: 3 dimensional tensor (B, H, W) \n 
              where B: batch size,   C: number of classes,   H: height,   W: width
            - smooth: for numerical stability
            """        
        recall_score = self.recall_score(prediction, ground_truth)
        precision_score = self.precision_score(prediction, ground_truth)        
        
        f_score = 2 * (recall_score * precision_score + smooth) / (recall_score + precision_score + smooth)
        
        return f_score