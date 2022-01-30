import torch
import torch.nn as nn
import torch.nn.functional as F



#MEAN IOU SCORE
class MeanIOUScore(nn.Module):
    """Computes IOU Score. \n
         Formula: (A ∩ B) / (A U B) \n
         A and B is predicted and ground truth values.
    """
    def __init__(self):
        super(MeanIOUScore, self).__init__()

    def forward(self, prediction, ground_truth, smooth=1e-6):
        """Args:
            - prediction: 4 dimensional tensor (B, C, H, W)
            - ground_truth: 3 dimensional tensor (B, H, W) \n 
              where B: batch size,   C: number of classes,   H: height,   W: width
            - smooth: for numerical stability
            """
        softmax_prediction = F.softmax(prediction, dim=1)
        argmax_prediction = torch.argmax(softmax_prediction, dim=1)

        intersections = (argmax_prediction == ground_truth).sum()
        unions = 2 * ground_truth.numel() - intersections

        iou = (intersections + smooth) / (unions + smooth)

        return iou



#PRECISION SCORE
class PrecisionScore(nn.Module):
    def __init__(self):
        """Computes Precision Score. \n
           Formula: TP / (TP + FP) \n
         - TP: True Positives 
         - FP: False Positives
        """
        super(PrecisionScore, self).__init__()

    def forward(self, prediction, ground_truth, smooth=1e-6):
        """Args:
            - prediction: 4 dimensional tensor (B, C, H, W)
            - ground_truth: 3 dimensional tensor (B, H, W) \n 
              where B: batch size,   C: number of classes,   H: height,   W: width
            - smooth: for numerical stability
            """
        classes = prediction.shape[1]
        softmax_prediction = F.softmax(prediction, dim=1)
        argmax_prediction = softmax_prediction.argmax(dim=1)

        score_list = [0] * classes

        for i in range(classes):
            TP = ((argmax_prediction == i) & (ground_truth == i)).sum()
            FP = (argmax_prediction == i).sum() - TP

            precision_score = (TP + smooth) / (TP + FP + smooth)
            score_list[i] = precision_score

        return score_list




#RECALL SCORE
class RecallScore(nn.Module):
    def __init__(self):
        """Computes Recall Score. \n
           Formula: TP / (TP + FN) \n
         - TP: True Positives 
         - FN: False Negatives
        """
        super(RecallScore, self).__init__()

    def forward(self, prediction, ground_truth, smooth=1e-6):
        """Args:
            - prediction: 4 dimensional tensor (B, C, H, W)
            - ground_truth: 3 dimensional tensor (B, H, W) \n 
              where B: batch size,   C: number of classes,   H: height,   W: width
            - smooth: for numerical stability
            """
        classes = prediction.shape[1]
        softmax_prediction = F.softmax(prediction, dim=1)
        argmax_prediction = softmax_prediction.argmax(dim=1)

        score_list = [0] * classes
        
        for i in range(classes):
            TP = ((argmax_prediction == i) & (ground_truth == i)).sum()
            FN = (argmax_prediction == i).sum() - TP

            recall_score = (TP + smooth) / (TP + FN + smooth)
            score_list[i] = recall_score
        
        return score_list



#F-SCORE
class FScore(nn.Module):
    def __init__(self):
        """Computes F Score. \n
           Formula: (2 * Recall * Precision) / (Recall + Precision) \n
        """
        super(FScore, self).__init__()
        self.recall_score = RecallScore()
        self.precision_score = PrecisionScore()

    def forward(self, prediction, ground_truth, smooth=1e-6):
        """Args:
            - prediction: 4 dimensional tensor (B, C, H, W)
            - ground_truth: 3 dimensional tensor (B, H, W) \n 
              where B: batch size,   C: number of classes,   H: height,   W: width
            - smooth: for numerical stability
            """
        classes = prediction.shape[1]
        score_list = [0] * classes
        
        recall_score = self.recall_score(prediction, ground_truth)
        precision_score = self.precision_score(prediction, ground_truth)        
        
        for i in range(classes):
            f_score = 2 * (recall_score[i] * precision_score[i] + smooth) / (recall_score[i] + precision_score[i] + smooth)
            score_list[i] = f_score
        
        return score_list
