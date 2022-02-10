import torch

#Data split ratios
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.70, 0.15, 0.15

#Training device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if DEVICE == "cuda" else False

#Input channel and number of classes
INPUT_CHANNELS = 4
NUM_CLASSES = 4

#Training number of epoch and batch size
NUM_EPOCHES = 20
BATCH_SIZE = 32

LR = 0.001