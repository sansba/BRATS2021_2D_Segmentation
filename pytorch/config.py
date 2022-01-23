import torch

#Data split ratios
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.70, 0.15, 0.15

#Training device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if DEVICE == "cuda" else False

#Input channel and number of classes
INPUT_CHANNELS = 1
NUM_CLASSES = 4

#Training number of epoch and batch size
NUM_EPOCHES = 30
BATCH_SIZE = 32

#Image size
INPUT_IMAGE_HEIGHT = 240
INPUT_IMAGE_WIDTH = 240

#Encoder-Decoder Channels
CHANNELS = [64, 128, 256, 512, 1024]

LR = 0.001