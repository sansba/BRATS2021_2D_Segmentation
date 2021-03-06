import torch
import numpy as np
import cv2

from matplotlib import pyplot as plt



def label_rgb_convert(label):
    """Converts gray label images to rgb images. \n
        "Args:
            - label (H x W): numpy uint8 image.
    """
    label[:, :, 0][(label[:, :, 0] == 0)] = 0
    label[:, :, 1][(label[:, :, 1] == 0)] = 0
    label[:, :, 2][(label[:, :, 2] == 0)] = 0

    label[:, :, 0][(label[:, :, 0] == 1)] = 255
    label[:, :, 1][(label[:, :, 1] == 1)] = 0
    label[:, :, 2][(label[:, :, 2] == 1)] = 0

    label[:, :, 0][(label[:, :, 0] == 2)] = 0
    label[:, :, 1][(label[:, :, 1] == 2)] = 255
    label[:, :, 2][(label[:, :, 2] == 2)] = 0

    label[:, :, 0][(label[:, :, 0] == 3)] = 0
    label[:, :, 1][(label[:, :, 1] == 3)] = 0
    label[:, :, 2][(label[:, :, 2] == 3)] = 255



def add_weights(input, label):
    """Add input and label images. \n
        "Args:
            - input (H x W): numpy uint8 image.
            - label (H x W): numpy uint8 image.
    """
    input = cv2.cvtColor(input, cv2.COLOR_GRAY2RGB)
    label = cv2.cvtColor(label, cv2.COLOR_GRAY2RGB)
    label_rgb_convert(label)

    output = cv2.addWeighted(input, 0.8, label, 1, 10)

    return output



def plot(input, ground_truth, prediction, n_rows, n_cols):
    input = input.cpu().numpy()
    input = np.uint8(input)

    ground_truth = ground_truth.cpu().numpy()
    ground_truth = np.uint8(ground_truth)

    prediction = torch.argmax(prediction, dim=1).cpu().numpy()
    prediction = np.uint8(prediction)

    fig, axis = plt.subplots(3, 4, figsize=(16, 12), sharex=True, sharey=True)
    fig.suptitle("Predicted Images on Test Data (Only Flair)")

    counter = 0
    for i in range(0, n_rows):
        for j in range(0, n_cols, 2):
            axis[i, j].title.set_text("Ground Truth")
            axis[i, j].imshow(add_weights(input[counter, 0], ground_truth[counter]))
            axis[i, j + 1].title.set_text("Predicted")
            axis[i, j + 1].imshow(add_weights(input[counter, 0], prediction[counter]))
            counter += 1



def history_plot(train_accum, val_accum):
    #LOSSES
    x_range_l = range(1, len(train_accum.all_losses[0]) + 1)
    len_criterion = len(train_accum.criterion_names)

    if len_criterion > 1:
        fig_criterion, axis_criterion = plt.subplots(1, len_criterion, sharex=True, figsize=(5 * len_criterion, 5))
        fig_criterion.suptitle("Criterion Losses per each epoch")
        for i in range(len_criterion):
            axis_criterion[i].title.set_text(train_accum.criterion_names[i])
            axis_criterion[i].plot(x_range_l, train_accum.all_losses[i])
            axis_criterion[i].plot(x_range_l, val_accum.all_losses[i])
            axis_criterion[i].legend(["Train", "Val"])

    else:
        plt.suptitle("Criterion Losses per each epoch")
        plt.title(train_accum.criterion_names[0])
        plt.plot(x_range_l, train_accum.all_losses[0])
        plt.plot(x_range_l, val_accum.all_losses[0])
        plt.legend(["Train", "Val"])
        plt.figure(figsize=(5, 5))

    
    #METRICS
    x_range_m = range(1, len(train_accum.all_metrics[0]) + 1)
    len_metrics = len(train_accum.metric_names)

    if len_metrics > 1:
        fig_metrics, axis_metrics = plt.subplots(1, len_metrics, sharex=True, figsize=(5 * len_metrics, 5))
        fig_metrics.suptitle("Metric Scores per each epoch")
        for i in range(len_metrics):
            axis_metrics[i].title.set_text(train_accum.metric_names[i])
            axis_metrics[i].plot(x_range_m, train_accum.all_metrics[i])
            axis_metrics[i].plot(x_range_m, val_accum.all_metrics[i])
            axis_metrics[i].legend(["Train", "Val"])

    else:
        plt.suptitle("Metric Scores per each epoch")
        plt.title(train_accum.metric_names[0])
        plt.plot(x_range_m, train_accum.all_metrics[0])
        plt.plot(x_range_m, val_accum.all_metrics[0])
        plt.legend(["Train", "Val"])
        plt.figure(figsize=(5, 5))