import torch.nn as nn
from torchsummary import summary

from model_blocks import *
import functional



#BASIC UNET MODEL
class UNet(nn.Module):
    def __init__(self, in_ch: int, n_classes: int):
        """Basic UNet Model. \n
            Args:
                - in_ch (int): input channel of input image.
                - n_classes (int): number of classes. 
        """
        super(UNet, self).__init__()
        self.in_ch = in_ch
        self.n_classes = n_classes
        
        self.inconv = InConv(in_ch, 64)
        
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)

        self.outconv = OutConv(64, n_classes)


    def forward(self, x):
        x1 = self.inconv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outconv(x)

        return x




#INCEPTION UNET MODEL
class InceptionUNet(nn.Module):
    def __init__(self, in_ch: int, n_classes: int):
        """Inception UNet Model. \n
            Args:
                - in_ch (int): input channel of input image.
                - n_classes (int): number of classes. 
        """
        super(InceptionUNet, self).__init__()
        self.in_ch = in_ch
        self.n_classes = n_classes

        self.inconv = IncInConv(in_ch, 64)

        self.down1 = IncDown(64, 128)
        self.down2 = IncDown(128, 256)
        self.down3 = IncDown(256, 512)
        self.down4 = IncDown(512, 1024)

        self.up1 = IncUp(1024, 512)
        self.up2 = IncUp(512, 256)
        self.up3 = IncUp(256, 128)
        self.up4 = IncUp(128, 64)

        self.outconv = IncOutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inconv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outconv(x)

        return x



#ARROW UNET MODEL
class ArrowUNet(nn.Module):
    def __init__(self, in_ch: int, n_classes: int):
        """Arrow UNet Model. \n
            Args:
                - in_ch (int): input channel of input image.
                - n_classes (int): number of classes. 
        """
        super(ArrowUNet, self).__init__()
        self.in_ch = in_ch
        self.n_classes = n_classes
        channel = 56

        self.inconv = ArrInConv(in_ch, channel)

        self.down1 = ArrDown(channel, channel * 2)
        self.down2 = ArrDown(channel * 2, channel * 4)
        self.down3 = ArrDown(channel * 4, channel * 8)
        self.down4 = ArrDown(channel * 8, channel * 16)

        self.up1 = ArrUp(channel * 16, channel * 8)
        self.up2 = ArrUp(channel * 8, channel * 4)
        self.up3 = ArrUp(channel * 4, channel * 2)
        self.up4 = ArrUp(channel * 2, channel)

        self.outconv = ArrOutConv(channel, n_classes)

    def forward(self, x):
        x1 = self.inconv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outconv(x)

        return x




#SEGMENTATIN MODELS
class SegmentationModels:
    def __init__(self, model_name, in_ch, n_classes):
        """General class for segmentation models. \n
            Args:
                - model_name (str): model selection ('unet', 'incunet', 'arrowunet').
                - in_ch (int): input channel of input image.
                - n_classes (int): number of classes.
        """
        self.models = {"unet":UNet, "inception":InceptionUNet, "arrow":ArrowUNet}
        self.model_name = model_name

        #Not In the List Error
        if not self.model_name.lower() in list(self.models.keys()):
            raise ValueError(f"There is no such a model called {self.model_name}. It must be 'unet', 'inception' or 'arrow'.")

        self.in_ch = in_ch
        self.n_classes = n_classes
        self.model = self.models[model_name.lower()](self.in_ch, self.n_classes)


        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        self.optimizer = None
        self.criterions = None
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
    def compile(self, epochs, optimizer, criterions, metrics, callbacks):
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
        if not hasattr(metrics, "__len__"):
            metrics = [metrics]

        if not hasattr(callbacks, "__len__"):
            callbacks = [callbacks]

        self.optimizer = optimizer
        self.criterions = criterions
        self.metrics = metrics
        self.callbacks = callbacks

        self.train_accum = functional.Accumulator(self.criterions, self.metrics)
        self.val_accum = functional.Accumulator(self.criterions, self.metrics)

        functional.train_net(self.train_loader, self.val_loader, self.train_accum, self.val_accum,  self.model, self.optimizer, self.criterions, self.metrics, self.callbacks, epochs, self.print)
        

    #Predict
    def predict(self, test_loader, path):
        """Predicts test data. \n
            Args:
                - test_loader (DataLoader): test dataset loader.
                - path (str): path where predictions will be saved.
        """
        self.test_loader = test_loader
        self.test_accum = functional.Accumulator(self.criterions)
        functional.predict_test(self.test_loader, self.test_accum, self.model, self.criterions, self.metrics, path)
    

    #Print
    def print(self):
        message_train = "TRAINING: "
        message_val = "VALIDATION: "
        for i, name in enumerate(self.train_accum.criterion_names):
            message_train += f"{name}: " + f"{self.train_accum.criterion_scores[i].float()}\t"
            message_val += f"{name}: " + f"{self.val_accum.criterion_scores[i].float()}\t"

        print(message_train, message_val, sep="\n")


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
