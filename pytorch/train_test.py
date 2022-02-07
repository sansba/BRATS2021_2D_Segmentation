import torch
import torch.nn as nn

from torch.utils.data import DataLoader

import config
import dataset

import models

import losses
import metrics
import callbacks



#Data preparation
datasets = dataset.BratsDataset("/content/drive/MyDrive/BRATS2021", n_data=5000)
datasets = dataset.train_val_test_split(datasets, test_val_split=0.15)

train_loader = DataLoader(datasets["train"], shuffle=True, batch_size=config.BATCH_SIZE)
val_loader = DataLoader(datasets["val"], shuffle=False, batch_size=config.BATCH_SIZE)
test_loader = DataLoader(datasets["test"], shuffle=False, batch_size=config.BATCH_SIZE)

#Model
model = models.SegmentationModels("unet", 1, 4)

#Criterion
ce_loss = nn.CrossEntropyLoss()

#Metric
iou_score = metrics.MeanIOUScore()
recall = metrics.RecallScore()
precision = metrics.PrecisionScore()

#Optimizer
optimizer = torch.optim.Adam(model.model.parameters(), config.LR)

#Callbacks
early_stop = callbacks.EarlyStopping(verbose=True, path="unet_early_model.pt")
best_model = callbacks.SaveBestModel(verbose=True, path="unet_best_model.pt")

#Fitting data into the model
model.fit(train_loader, val_loader)

#Training the model
model.compile(epochs=config.NUM_EPOCHES, optimizer=optimizer, criterions=[ce_loss], metrics=[iou_score, recall, precision], callbacks=[early_stop, best_model])


model.predict(test_loader)