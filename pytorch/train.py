import time

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

import config
import dataset
import models
import utils
import losses

import numpy as np
from tqdm import tqdm




#TRAIN EPOCH
def train_epoch(net, train_loader, criterion, metric, optimizer):
    """Trains single epoch."""
    net.train()
    train_loss = 0
    train_acc = 0

    for image, label in tqdm(train_loader):
        image = image.to(config.DEVICE)
        label = label.to(config.DEVICE)


        prediction = net(image)
        loss = criterion(prediction, label)
        accuracy = metric(prediction, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc += accuracy
        train_loss += loss.item() / image.shape[0]

    return [train_loss, train_acc]



#Validation of epoch
def validate_epoch(net, val_loader, criterion, metric):
    net.eval()
    val_loss = 0
    val_acc = 0

    for image, label in val_loader:
        with torch.no_grad():
            image = image.to(config.DEVICE)
            label = label.to(config.DEVICE)

            prediction = net(image)
            loss = criterion(prediction, label)
            accuracy = metric(prediction, label)
            
            val_acc += accuracy
            val_loss += loss.item() / image.shape[0]

    return [val_loss, val_acc]
            



#TRAINING NET
def train_net(train_loader, val_loader, net, criterion, metric, optimizer, num_epochs=10, batch_size=32):
    """Trains model as number of epochs as model have."""
    val_loss = np.inf

    start = time.time()

    net.apply(utils.init_weights)

    for epoch in range(num_epochs):
        train_metrics = train_epoch(net, train_loader, criterion, metric, optimizer)
        val_metrics = validate_epoch(net, val_loader, criterion, metric)
        
        print("Epoch", epoch + 1)
        print("Train Loss: {}, Train Accuracy: {} ".format(train_metrics[0], train_metrics[1]))
        print("Validation Loss: {}, Validation Accuracy: {} ".format(val_metrics[0], val_metrics[1]))
        utils.save_best_model(net, val_loss, val_metrics[0])

    print("Geçen süre: ", time.time() - start)


#Data preparation
datasets = dataset.BratsDataset("brats2021")
datasets = dataset.train_val_test_dataset(datasets)
train_loader = DataLoader(datasets["train"].dataset, shuffle=True, batch_size=config.BATCH_SIZE)
val_loader = DataLoader(datasets["val"].dataset, shuffle=False, batch_size=config.BATCH_SIZE)

#Model
unet = models.ArrowUNet(1, 4).to(config.DEVICE)

#Loss function metric optimizer
criterion = nn.CrossEntropyLoss()
metric = losses.MeanIOU()
optimizer = torch.optim.Adam(unet.parameters(), config.LR)

#Training model
train_net(train_loader, val_loader, unet, criterion, metric, optimizer, num_epochs=1, batch_size=config.BATCH_SIZE)