import torch
from . import utils


def predict_test(test_loader, net, criterion, n=6):
    """Predicting entire dataset and accuracy and plotting n images and their labels"""
    net.eval()
    total_loss = 0
    total_iou = 0
    
    with torch.no_grad():
        for image, label in test_loader:
            prediction = net(image)

            iou_acc = utils.MeanIOU(prediction, label)
            loss = criterion(prediction, label)

            total_iou += iou_acc
            total_loss += loss

    return total_iou, total_loss