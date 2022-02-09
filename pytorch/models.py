import torch.nn as nn
from torchsummary import summary

import config

from model_blocks import *
import functional
import plot


#BASIC UNET MODEL
class UNet(nn.Module):
    def __init__(self, in_ch, n_classes):
        """Basic UNet Model. \n
            Args:
                - in_ch (int): input channel of input image.
                - n_classes (int): number of classes. 
        """
        super(UNet, self).__init__()
        self.in_ch = in_ch
        self.n_classes = n_classes
        channel = 64
        
        self.inconv = InConv(in_ch, channel)
        
        self.down1 = Down(channel, channel * 2)
        self.down2 = Down(channel * 2, channel * 4)
        self.down3 = Down(channel * 4, channel * 8)
        #self.down4 = Down(channel * 8, channel * 16)

        #self.up1 = Up(channel * 16, channel * 8)
        self.up2 = Up(channel * 8, channel * 4)
        self.up3 = Up(channel * 4, channel * 2)
        self.up4 = Up(channel * 2, channel)

        self.outconv = OutConv(channel, n_classes)


    def forward(self, x):
        x1 = self.inconv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        #x5 = self.down4(x4)

        #x = self.up1(x5, x4)
        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outconv(x)

        return x




#INCEPTION UNET MODEL
class InceptionUNet(nn.Module):
    def __init__(self, in_ch, n_classes):
        """Inception UNet Model. \n
            Args:
                - in_ch (int): input channel of input image.
                - n_classes (int): number of classes. 
        """
        super(InceptionUNet, self).__init__()
        self.in_ch = in_ch
        self.n_classes = n_classes
        channel = 64

        self.inconv = IncInConv(in_ch, channel)

        self.down1 = IncDown(channel, channel * 2)
        self.down2 = IncDown(channel * 2, channel * 4)
        self.down3 = IncDown(channel * 4, channel * 8)
        #self.down4 = IncDown(channel * 8, channel * 16)

        #self.up1 = IncUp(channel * 16, channel * 8)
        self.up2 = IncUp(channel * 8, channel * 4)
        self.up3 = IncUp(channel * 4, channel * 2)
        self.up4 = IncUp(channel * 2, channel)

        self.outconv = IncOutConv(channel, n_classes)

    def forward(self, x):
        x1 = self.inconv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        #x5 = self.down4(x4)

        #x = self.up1(x5, x4)
        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outconv(x)

        return x



#ARROW UNET MODEL
class ArrowUNet(nn.Module):
    def __init__(self, in_ch, n_classes):
        """Arrow UNet Model. \n
            Args:
                - in_ch (int): input channel of input image.
                - n_classes (int): number of classes. 
        """
        super(ArrowUNet, self).__init__()
        self.in_ch = in_ch
        self.n_classes = n_classes
        channel = 63

        self.inconv = ArrInConv(in_ch, channel)

        self.down1 = ArrDown(channel, channel * 2)
        self.down2 = ArrDown(channel * 2, channel * 4)
        self.down3 = ArrDown(channel * 4, channel * 8)
        #self.down4 = ArrDown(channel * 8, channel * 16)

        #self.up1 = ArrUp(channel * 16, channel * 8)
        self.up2 = ArrUp(channel * 8, channel * 4)
        self.up3 = ArrUp(channel * 4, channel * 2)
        self.up4 = ArrUp(channel * 2, channel)

        self.outconv = ArrOutConv(channel, n_classes)

    def forward(self, x):
        x1 = self.inconv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        #x5 = self.down4(x4)

        #x = self.up1(x5, x4)
        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outconv(x)

        return x




#SEGMENTATIN MODELS
class SegmentationModels:
    def __init__(self, model_name, in_ch, n_classes, L=4):
        """General class for segmentation models. \n
            Args:
                - model_name (str): model selection ('unet', 'incunet', 'arrowunet').
                - in_ch (int): input channel of input image.
                - n_classes (int): number of classes.
                - L (int): deepness of model
        """
        self.models = {"unet":UNet, "inception":InceptionUNet, "arrow":ArrowUNet}
        self.model_name = model_name

        #Not In the List Error
        if not self.model_name.lower() in list(self.models.keys()):
            raise ValueError(f"There is no such a model called {self.model_name}. It must be 'unet', 'inception' or 'arrow'.")

        self.in_ch = in_ch
        self.n_classes = n_classes
        self.L = L
        self.model = self.models[model_name.lower()](self.in_ch, self.n_classes, self.L).to(config.DEVICE)


        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        self.optimizer = None
        self.criterion = None
        self.metrics = None
        self.callbacks = None


    #Fit
    def fit(self, train_loader, val_loader):
        """Fits datasets to model. \n
           Args:
                - train_loader (DataLoader): training dataset loader.
                - val_loader (DataLoader): validation dataset loader.
        """
        self.train_loader = train_loader
        self.val_loader = val_loader


    #Compile
    def compile(self, epochs, optimizer, criterion, metrics, callbacks, is_init=True):
        """Trains model. \n
            Args:
                - epochs (int): number of epochs.
                - batch_size (int): size of each data iter.
                - optimizer : model optimizer.
                - criterion : loss function.
            Examples:
                >>> model = Model()
                >>> optimizer = torch.nn.optim.SGD(model.parameters(), lr=0.001)
                >>> criterion = torch.nn.CrossEntropyLoss()
                >>> model.fit(train_loader, val_loader)
                >>> model.compile(20, 32, optimizer, criterion)
        """
        if not hasattr(criterion, "__len__"):
            criterion = [criterion]
        
        if not hasattr(metrics, "__len__"):
            metrics = [metrics]

        if not hasattr(callbacks, "__len__"):
            callbacks = [callbacks]

        self.optimizer = optimizer
        self.criterion = criterion
        self.metrics = metrics
        self.callbacks = callbacks

        self.train_accum = functional.Accumulator(self.criterion, self.metrics)
        self.val_accum = functional.Accumulator(self.criterion, self.metrics)

        functional.train_net(self.train_loader, self.val_loader, self.train_accum, self.val_accum,  self.model, self.optimizer, self.criterion, self.metrics, self.callbacks, epochs, self.training_print, is_init)
        

    #Predict
    def predict(self, test_loader):
        """Predicts test data. \n
            Args:
                - test_loader (DataLoader): test dataset loader.
                - path (str): path where predictions will be saved.
        """
        self.test_loader = test_loader
        self.test_accum = functional.Accumulator(self.criterion, self.metrics)

        functional.predict_test(self.test_loader, self.test_accum, self.model, self.criterion, self.metrics)
        self.test_print()
    

    #Training Print
    def training_print(self):
        message_train = "TRAINING: "
        message_val = "VALIDATION: "

        for name, train_score, val_score in zip(self.train_accum.criterion_names, self.train_accum.criterion_scores, self.val_accum.criterion_scores):
            message_train += f"{name}: {train_score} \t"
            message_val += f"{name}: {val_score} \t"

        for name, train_score, val_score in zip(self.train_accum.metric_names, self.train_accum.metric_scores, self.val_accum.metric_scores):
            message_train += f"{name}: {train_score} \t"
            message_val += f"{name}: {val_score} \t"

        print(message_train, message_val, sep="\n")


    #Test Print
    def test_print(self):
        message_test = "TEST: "

        for name, test_score in zip(self.test_accum.criterion_names, self.test_accum.criterion_scores):
            message_test += f"{name}: {test_score} \t"

        for name, test_score in zip(self.test_accum.metric_names, self.test_accum.metric_scores):
            message_test += f"{name}: {test_score} \t"

        print(message_test)


    #History
    def history(self):
        plot.history_plot(self.test_accum, self.val_accum)
        

    #Param Upload
    def param_upload(self, path):
        """Uploads the parameters of a model. \n
            Args:
                - path (str): path to a pt file.
        """
        self.model.state_dict(torch.load(path))


    #Summary
    def summary(self, input_size):
        """Prints model summary.\n
            Args:
                - inputs_size (tuple): (H, W) \n
                where H: height,  W: width
        """
        input_size = (self.in_ch, ) + input_size
        summary(self.model, input_size)
        print("Model Name: ", self.model.__class__.__name__)