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
        self.down4 = Down(channel * 8, channel * 16)

        self.up1 = Up(channel * 16, channel * 8)
        self.up2 = Up(channel * 8, channel * 4)
        self.up3 = Up(channel * 4, channel * 2)
        self.up4 = Up(channel * 2, channel)

        self.outconv = OutConv(channel, n_classes)


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



#EXPANDED UNET MODEL
class ExpandedUNet(nn.Module):
    def __init__(self, in_ch, n_classes):
        """Basic UNet Model. \n
            Args:
                - in_ch (int): input channel of input image.
                - n_classes (int): number of classes. 
        """
        super(ExpandedUNet, self).__init__()
        self.in_ch = in_ch
        self.n_classes = n_classes
        channel = 64
        
        self.inconv = ExpInConv(in_ch, channel)
        
        self.down1 = ExpDown(channel, channel * 2)
        self.down2 = ExpDown(channel * 2, channel * 4)
        self.down3 = ExpDown(channel * 4, channel * 8)
        self.down4 = ExpDown(channel * 8, channel * 16)

        self.up1 = ExpUp(channel * 16, channel * 8)
        self.up2 = ExpUp(channel * 8, channel * 4)
        self.up3 = ExpUp(channel * 4, channel * 2)
        self.up4 = ExpUp(channel * 2, channel)

        self.outconv = OutConv(channel, n_classes)


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
        self.down4 = IncDown(channel * 8, channel * 16)

        self.up1 = IncUp(channel * 16, channel * 8)
        self.up2 = IncUp(channel * 8, channel * 4)
        self.up3 = IncUp(channel * 4, channel * 2)
        self.up4 = IncUp(channel * 2, channel)

        self.outconv = IncOutConv(channel, n_classes)

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
    def __init__(self, in_ch, n_classes):
        """Arrow UNet Model. \n
            Args:
                - in_ch (int): input channel of input image.
                - n_classes (int): number of classes. 
        """
        super(ArrowUNet, self).__init__()
        self.in_ch = in_ch
        self.n_classes = n_classes
        channel = 64

        self.inconv = InConv(in_ch, channel)

        self.down1 = ArrDown(channel, 2)
        self.down2 = ArrDown(channel * 2, 3)
        self.down3 = ArrDown(channel * 3, 4)
        self.down4 = ArrDown(channel * 4, 5)

        self.up1 = ArrUp(channel * 5, 4)
        self.up2 = ArrUp(channel * 6, 3)
        self.up3 = ArrUp(channel * 7, 2)
        self.up4 = ArrUp(channel * 8, 1)

        self.outconv = OutConv(channel * 9, n_classes)

    def forward(self, x):
        x1_en = self.inconv(x)
        x2_en = self.down1(x1_en)
        x3_en = self.down2(x1_en, x2_en)
        x4_en = self.down3(x1_en, x2_en, x3_en)
        x5_en = self.down4(x1_en, x2_en, x3_en, x4_en)

        x4_de = self.up1(x4_en, x5_en)
        x3_de = self.up2(x3_en, x5_en, x4_de)
        x2_de = self.up3(x2_en, x5_en, x4_de, x3_de)
        x1_de = self.up4(x1_en, x5_en, x4_de, x3_de, x2_de)
        
        return self.outconv(x1_de)



#UNET 3+ MODEL
class UNet3P(nn.Module):
    def __init__(self, in_ch, n_classes):
        """Arrow UNet Model. \n
            Args:
                - in_ch (int): input channel of input image.
                - n_classes (int): number of classes. 
        """
        super(UNet3P, self).__init__()
        self.in_ch = in_ch
        self.n_classes = n_classes
        self.channel = 64
        
        self.inconv = InConv(in_ch, self.channel)
        
        self.down1 = Down(self.channel, self.channel * 2)
        self.down2 = Down(self.channel * 2, self.channel * 4)
        self.down3 = Down(self.channel * 4, self.channel * 8)
        self.down4 = Down(self.channel * 8, self.channel * 16)


        self.up1 = ThreeUp(4)
        self.up2 = ThreeUp(3)
        self.up3 = ThreeUp(2)
        self.up4 = ThreeUp(1)

        self.outconv = ThreeOut(self.channel, n_classes)
        self.outconv1 = nn.Sequential(nn.Conv2d(in_channels=1024, out_channels=n_classes, kernel_size=1), nn.AdaptiveMaxPool2d(160), nn.Sigmoid())


    def forward(self, x):
        x1_en = self.inconv(x)
        x2_en = self.down1(x1_en)
        x3_en = self.down2(x2_en)
        x4_en = self.down3(x3_en)
        x5_en = self.down4(x4_en)

        x4_de = self.up1(x1_en, x2_en, x3_en, x4_en, x5_en)
        x3_de = self.up2(x1_en, x2_en, x3_en, x4_de, x5_en)
        x2_de = self.up3(x1_en, x2_en, x3_de, x4_de, x5_en)
        x1_de = self.up4(x1_en, x2_de, x3_de, x4_de, x5_en)

        out1 = self.outconv(x1_de)
        out2 = self.outconv(x2_de)
        out3 = self.outconv(x3_de)
        out4 = self.outconv(x4_de)
        out5 = self.outconv1(x5_en)

        x1 = torch.mean(torch.cat([out1[:, 0].unsqueeze(1), out2[:, 0].unsqueeze(1), out3[:, 0].unsqueeze(1), out4[:, 0].unsqueeze(1), out5[:, 0].unsqueeze(1)], dim=1), dim=1).unsqueeze(1)
        x2 = torch.mean(torch.cat([out1[:, 1].unsqueeze(1), out2[:, 1].unsqueeze(1), out3[:, 1].unsqueeze(1), out4[:, 1].unsqueeze(1), out5[:, 1].unsqueeze(1)], dim=1), dim=1).unsqueeze(1)
        x3 = torch.mean(torch.cat([out1[:, 2].unsqueeze(1), out2[:, 2].unsqueeze(1), out3[:, 2].unsqueeze(1), out4[:, 2].unsqueeze(1), out5[:, 2].unsqueeze(1)], dim=1), dim=1).unsqueeze(1)
        x4 = torch.mean(torch.cat([out1[:, 3].unsqueeze(1), out2[:, 3].unsqueeze(1), out3[:, 3].unsqueeze(1), out4[:, 3].unsqueeze(1), out5[:, 3].unsqueeze(1)], dim=1), dim=1).unsqueeze(1)

        return torch.cat([x1, x2, x3, x4], dim=1)




#SEGMENTATIN MODELS
class SegmentationModels:
    def __init__(self, model_name, in_ch, n_classes):
        """General class for segmentation models. \n
            Args:
                - model_name (str): model selection ('unet', 'expanded', 'inception', 'arrow', 'unet3+').
                - in_ch (int): input channel of input image.
                - n_classes (int): number of classes.
                - L (int): deepness of model
        """
        self.models = {"unet":UNet, "expanded":ExpandedUNet, "inception":InceptionUNet, "arrow":ArrowUNet, "unet3+":UNet3P}
        self.model_name = model_name

        #Not In the List Error
        if not self.model_name.lower() in list(self.models.keys()):
            raise ValueError(f"There is no such a model called {self.model_name}. It must be 'unet', 'inception' or 'arrow'.")

        self.in_ch = in_ch
        self.n_classes = n_classes
        self.model = self.models[model_name.lower()](self.in_ch, self.n_classes).to(config.DEVICE)


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