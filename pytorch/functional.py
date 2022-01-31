import torch
import torch.nn as nn

from tqdm import tqdm
import config



#Train Epoch
def train_epoch(train_loader, accumulator, model, optimizer, criterions, metrics):
    """Trains single epoch. \n
        Args:
            - train_loader (DataLoader): train dataset loader.
            - accumulator (Accumulator): accumulator for losses and metrics.
            - model (Model): segmentation model.
            - criterions: loss function.
            - optimizer: optimization function.
            - metrics: metrics for model's evaluation.
    """
    model.train()
    accumulator.reset()
    
    for image, label in tqdm(train_loader):
        prediction = model(image)
        
        losses = 0
        for i, loss in enumerate(criterions):
            l = loss(prediction, label)
            losses += l
            accumulator.add_losses(l.item(), i)
        losses = losses / len(criterions)
        
        for i, metric in enumerate(metrics):
            accumulator.add_metrics(metric(prediction, label).item(), i)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()



#Validate Epoch
def validate_epoch(val_loader, accumulator, model, criterions, metrics):
    """Trains single epoch. \n
        Args:
            - val_loader (DataLoader): validation dataset loader.
            - accumulator (Accumulator): accumulator for losses and metrics.
            - model (Model): segmentation model.
            - criterions: loss function.
            - metrics: metrics for model's evaluation.
    """
    model.eval()
    accumulator.reset()

    for image, label in val_loader:
        with torch.no_grad():
            prediction = model(image)

        for i, loss in enumerate(criterions):
            accumulator.add_losses(loss(prediction, label).item(), i)
        
        for i, metric in enumerate(metrics):
            accumulator.add_metrics(metric(prediction, label).item(), i)



#Train Net
def train_net(train_loader, val_loader, train_accum, val_accum, model, optimizer, criterions, metrics, callbacks, num_epochs, plot_fn):
    """Trains model as long as number of epochs. \n
        Args:
            - train_loader (DataLoader): training dataset loader.
            - val_loader (DataLoader): validation dataset loader.
            - train_accum (Accumulator): accumulator for train scores.
            - val_accum (Accumulator): accumulator for validation scores.
            - model (Model): segmentation model.
            - optimizer: optimization function.
            - criterions: loss function.
            - metrics: metrics for model's evaluation.
            - num_epochs (int): number of epochs.
            - plot_fn: prints train's and validation's losses and metrics.
    """
    model.apply(init_weights)

    for epoch in range(num_epochs):
        train_epoch(train_loader, train_accum, model, optimizer, criterions, metrics)
        validate_epoch(val_loader, val_accum, model, criterions, metrics)
        
        print(f"Epoch {epoch + 1}")
        plot_fn()
        for callback in callbacks:
            callback(val_accum.criterion_scores, model)



#Predict Test
def predict_test(test_loader, accumulator, model, criterions, metrics, path=None):
    """Predicts test dataset. \n
        Args:
            - test_loader (DataLoader): test dataset loader.
            - accumulator (Accumulator): accumulator for losses and metrics.
            - model (Model): segmentation model.
            - criterions: loss functions.
            - metrics: metric functions.
            - path (str): path where predicted images will be saved.
    """
    model.eval()
    accumulator.reset()
    
    with torch.no_grad():
        for image, label in test_loader:
            image = image.to(config.DEVICE)
            label = label.to(config.DEVICE)
            prediction = model(image).to(config.DEVICE)

        for i, loss in enumerate(criterions):
            accumulator.add_losses(loss(prediction, label), i)
        
        for i, metric in enumerate(metrics):
            accumulator.add_metrics(metric(prediction, label), i)



#Init Weight
def init_weights(m):
    """Initialization function for model parameters.
        Examples:
            >>> model = UNet()
            >>> model.apply(init_weights)
            """
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)



#Is Torch Integer
def is_torch_integer(input: torch.Tensor) -> bool:
    """Checks if tensor is a integer or not. \n
        Args:
            - input (Tensor): input tensor to be checked.
    """ 
    int_list = [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]
    
    for int_val in int_list:
        if input.dtype == int_val:
            return True
    return False



#ONE HOT ENCODER
class OneHotEncoder:
    def __init__(self):
        """One hot encoder transformer for N classes.
        Shape:
            - Input: (B, H, W) where 
                B = batch size of tensor, \n
                H = height of tensor image, \n
                W = width of tensor image.
        """

    def __call__(self, input: torch.Tensor, n_classes) -> torch.Tensor:
        #Dimension Error
        if input.ndimension() != 3:
            raise ValueError("Input tensor must be 3 dimensional. Got {} dimension".format(input.ndimension()))
        #Not Integer Error
        if not is_torch_integer(input):
            raise ValueError("Input tensor data type must be integer. Got {}".format(input.dtype))
        #Label Error
        if (input >= n_classes).any():
            raise ValueError("Values of input tensor must be between 0 and n_classes (0-C).")

        data_type = input.dtype
        batch_size = input.shape[0]
        height = input.shape[1]
        width = input.shape[2]
        
        output = torch.zeros((batch_size, n_classes, height, width), dtype=data_type)

        for i in range(n_classes):
            output[:, i, :, :][input == i] = 1

        return output



#Accumulator
class Accumulator:
    def __init__(self, criterions, metrics):
        """Accumulates train's and validation's loss values and metric scores for each iter. \n
            Args:
                - criterions (list): criterion's functions list.
                - metrics (list): metric's functions list.
        """
        self.criterion_scores = [0.0] * len(criterions)
        self.metric_scores = [0.0] * len(metrics)
        self.criterion_names = self.get_names(criterions)
        self.metric_names = self.get_names(metrics)

    #Add Losses
    def add_losses(self, criterion_scores, index):
        self.criterion_scores[index] += criterion_scores[index]

    #Add Metrics
    def add_metrics(self, metric_scores, index):
        self.metric_scores[index] += metric_scores[index]

    #Reset
    def reset(self):
        self.criterion_scores = [0.0] * len(self.criterion_scores)
        self.metric_scores = [0.0] * len(self.metric_scores)

    #Create Dict
    def get_names(self, input):
        """Gets criterions' and metrics' function name. \n
            Args:
                - input (list): input list.
        """
        names = []
        for val in input:
            name = val.__class__.__name__
            names += [name]
        return names
