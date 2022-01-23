import torch.nn as nn

from model_blocks import *
from torchsummary import summary




#BASIC UNET MODEL
class UNet(nn.Module):
    def __init__(self, in_ch, n_classes):
        super().__init__()
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

    def summary(self, input_size):
        print(summary(UNet(self.in_ch, self.n_classes), input_size))




#INCEPTION UNET MODEL
class IncUNet(nn.Module):
    def __init__(self, in_ch, n_classes):
        super().__init__()
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

    def summary(self, input_size):
        print(summary(IncUNet(self.in_ch, self.n_classes), input_size))




#ARROW UNET MODEL
class ArrowUNet(nn.Module):
    def __init__(self, in_ch, n_classes, channel=56):
        super().__init__()
        self.in_ch = in_ch
        self.n_classes = n_classes

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

    def summary(self, input_size):
        print(summary(ArrowUNet(self.in_ch, self.n_classes), input_size))