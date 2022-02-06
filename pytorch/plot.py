import torch
import numpy as np
import cv2

from matplotlib import pyplot as plt




def label_rgb_convert(label):
    """Converts gray label images to rgb images. \n
        "Args:
            - label (H x W): numpy uint8 image.
    """
    label[:, :, 0][(label[:, :, 0] == 1)] = 255
    label[:, :, 1][(label[:, :, 1] == 1)] = 0
    label[:, :, 2][(label[:, :, 2] == 1)] = 0

    label[:, :, 0][(label[:, :, 0] == 2)] = 0
    label[:, :, 1][(label[:, :, 1] == 2)] = 255
    label[:, :, 2][(label[:, :, 2] == 2)] = 0

    label[:, :, 0][(label[:, :, 0] == 4)] = 0
    label[:, :, 1][(label[:, :, 1] == 4)] = 0
    label[:, :, 2][(label[:, :, 2] == 4)] = 255



def add_weights(input, label):
    """Add input and label images. \n
        "Args:
            - input (H x W): numpy uint8 image.
            - label (H x W): numpy uint8 image.
    """
    input = cv2.cvtColor(input, cv2.COLOR_GRAY2RGB)
    label = label_rgb_convert(cv2.cvtColor(label, cv2.COLOR_GRAY2RGB))

    output = cv2.addWeighted(input, 0.8, label, 1, 10)

    return output





def plot(input, ground_truth, prediction, n_rows, n_cols):
    input = input.numpy()
    input = np.uint8(input)

    ground_truth = ground_truth.numpy()
    ground_truth = np.uint8(ground_truth)

    prediction = torch.argmax(prediction, dim=1).numpy()
    prediction = np.uint8(prediction)

    fig, axis = plt.subplots(3, 4, figsize=(16, 12), sharex=True, sharey=True)
    fig.suptitle("Predicted Images on Test Data")

    counter = 0
    for i in range(0, n_rows):
        for j in range(0, n_cols, 2):
            axis[i, j].title.set_text("Ground Truth")
            axis[i, j].imshow(add_weights(input[counter], ground_truth[counter]))
            axis[i, j + 1].title.set_text("Predicted")
            axis[i, j + 1].imshow(add_weights(input[counter], prediction[counter]))
            counter += 1
