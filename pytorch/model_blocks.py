import re
import torch
import torch.nn as nn
import torch.nn.functional as F


""" BASIC BLOCKS"""
#Double Conv
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        """Double Convolution for Basic UNet Model. \n
           Args:
           - in_ch (int): input channel of the first convolution layer.
           - out_ch (int): output channel of the first and second convolution layer and input channel of the second convolution layer.
        """
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding="same"), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                        nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding="same"), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


#Input Conv
class InConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        """Input Convolution Layer for Basic UNet Model. \n
           Args:
           - in_ch (int): input channel of the first convolution layer.
           - out_ch (int): output channel of the first and second convolution layer and input channel of the second convolution layer.
        """
        super(InConv, self).__init__()
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(x)


#Down Sampling
class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        """Down Sampling Layer for Basic UNet Model. \n
           Args:
           - in_ch (int): input channel of first convolution layer.
           - out_ch (int): output channel of the first and second convolution layer and input channel of the second convolution layer.
        """
        super(Down, self).__init__()
        self.down = nn.Sequential(nn.MaxPool2d(2, 2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.down(x)


#Up Sampling
class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        """Up Sampling Layer for Basic UNet Model. \n
           Args:
           - in_ch (int): input channel of first convolution layer.
           - out_ch (int): output channel of the first and second convolution layer and input channel of the second convolution layer.
        """
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels=in_ch, out_channels=in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x2 = F.pad(x2, (-(diffX // 2), -(diffX - diffX // 2),
                        -(diffY // 2), -(diffY - diffY // 2)))

        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)


#Out Conv
class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        """Out Convolution Layer for Basic UNet Model. \n
           Args:
           - in_ch (int): input channel of the convolution layer.
           - out_ch (int): output channel of the convolution layer.
        """
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)





""" INCEPTION BLOCKS"""
#Inception Block
class InceptionBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        """Inception Block for Inception UNet Model. \n
           Args:
           - in_ch (int): input channel of the convolution layer.
           - out_ch (int): output channel of the convolution layer.
        """
        super(InceptionBlock, self).__init__()
        self.x1 = nn.Sequential(nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding="same", dilation=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.x2 = nn.Sequential(nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding="same", dilation=2), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.x3 = nn.Sequential(nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding="same", dilation=3), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.x4 = nn.Sequential(nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding="same", dilation=4), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        x1 = self.x1(x)
        x2 = self.x2(x)
        x3 = self.x3(x)
        x4 = self.x4(x)
        x = torch.cat([x1, x2, x3, x4], dim=1)
        return x


#Double Conv
class IncDoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        """Double Inception Convolution for Inception UNet Model. \n
           Args:
           - in_ch (int): input channel of the first convolution layer.
           - out_ch (int): output channel of the first and second convolution layer and input channel of the second convolution layer.
        """
        super(IncDoubleConv, self).__init__()
        self.conv = nn.Sequential(InceptionBlock(in_ch, out_ch // 4), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                                InceptionBlock(out_ch, out_ch // 4), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


#Input Convolution
class IncInConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        """Input Convolution Layer for Inception UNet Model. \n
           Args:
           - in_ch (int): input channel of the convolution layer.
           - out_ch (int): output channel of the first and second convolution layer and input channel of the second convolution layer.
        """
        super(IncInConv, self).__init__()
        self.conv = IncDoubleConv(in_ch, out_ch)
    
    def forward(self, x):
        return self.conv(x)


#Down Sampling
class IncDown(nn.Module):
    def __init__(self, in_ch, out_ch):
        """Down Sampling Layer for Inception UNet Model. \n
           Args:
           - in_ch (int): input channel of the convolution layer.
           - out_ch (int): output channel of the first and second convolution layer and input channel of the second convolution layer.
        """
        super(IncDown, self).__init__()
        self.down = nn.Sequential(nn.MaxPool2d(2, 2), IncDoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.down(x)


#Up Sampling
class IncUp(nn.Module):
    def __init__(self, in_ch, out_ch):
        """Up Sampling Layer for Inception UNet Model. \n
           Args:
           - in_ch (int): input channel of the convolution layer.
           - out_ch (int): output channel of the first and second convolution layer and input channel of the second convolution layer.
        """
        super(IncUp, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels=in_ch, out_channels=in_ch // 2, kernel_size=2, stride=2)
        self.conv = IncDoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffX = x2.size()[3] - x1.size()[3]
        diffY = x2.size()[2] - x1.size()[2]

        x2 = F.pad(x2, (-(diffX // 2), -(diffX - diffX // 2),
                        -(diffY // 2), -(diffY - diffY // 2)))
        x = torch.cat([x2, x1], dim=1)
        
        return self.conv(x)


#Out Convolution
class IncOutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        """Output Convolution Layer for Inception UNet Model. \n
           Args:
           - in_ch (int): input channel of the convolution layer.
           - out_ch (int): output channel of the convolution layer.
        """
        super(IncOutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1)
        
    def forward(self, x):
        return self.conv(x)




""" ARROW BLOCKS """
#Double Conv
class ArrDoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        """Double Convolution for Basic UNet Model. \n
           Args:
           - in_ch (int): input channel of the first convolution layer.
           - out_ch (int): output channel of the first and second convolution layer and input channel of the second convolution layer.
        """
        super(ArrDoubleConv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding="same"), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                        nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding="same"))

    def forward(self, x):
        return self.conv(x)

#Down Sampling
class ArrDown(nn.Module):
    def __init__(self, in_ch, depth):
        """Down Sampling  Layer for Arrow UNet Model. \n
           Args:
           - in_ch (int): input channel of the convolution layer.
           - out_ch (int): output channel of the convolution layer.
        """
        super(ArrDown, self).__init__()
        self.depth = depth
        channel = 64

        self.down_list = []
        self.down = nn.Sequential(nn.MaxPool2d(2, 2), ArrDoubleConv(in_ch, channel))
        self.conv_out = nn.Sequential(nn.BatchNorm2d(channel * depth), nn.ReLU(inplace=True))

        for i in range(self.depth - 1):
            self.down_list.append(nn.Sequential(nn.MaxPool2d(2 ** (self.depth - i - 1)), nn.Conv2d(in_channels=channel * (i + 1), out_channels=64, kernel_size=3, padding="same")))


    def forward(self, *args):
        out = []

        for i in range(len(self.down_list)):
            x = self.down_list[i](args[i])
            out.append(x)

        out.append(self.down(args[-1]))
        x = torch.cat(out, dim=1)

        return self.conv_out(x)



#Up Sampling
class ArrUp(nn.Module):
    def __init__(self, in_ch, depth):
        """Up Sampling Layer for Arrow UNet Model. \n
           Args:
           - in_ch (int): input channel of the convolution layer.
           - out_ch (int): output channel of the convolution layer.
        """
        super(ArrUp, self).__init__()
        self.depth = depth
        channel = 64

        self.up_list = []
        self.up = nn.Sequential(nn.ConvTranspose2d(in_channels=in_ch, out_channels=channel * 4, kernel_size=2, stride=2), ArrDoubleConv(channel * 4, channel * 4))
        self.conv = nn.Conv2d(in_channels=channel * depth, out_channels=channel, kernel_size=3, padding="same")
        self.conv_out = nn.Sequential(nn.BatchNorm2d(channel * (10 - depth)), nn.ReLU(inplace=True))

        for i in range(5, self.depth, -1):
            self.up_list.append(nn.Sequential(nn.ConvTranspose2d(in_channels=channel * (10 - i), out_channels=channel, kernel_size=2 ** (i - self.depth), stride=2 ** (i - self.depth)), nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, padding="same")))


    def forward(self, *args):
        out = [self.conv(args[0])]

        for i in range(len(self.up_list)):
            out.append(self.up_list[i](args[i + 1]))
        
        out.append(self.up(args[-1]))
        x = torch.cat(out, dim=1)
        
        return self.conv_out(x)
        




""" DENSE BLOCKS """
class ThreeUp(nn.Module):
    def __init__(self, depth):
        """Dense Block for Dense UNet Model. \n
           Args:
           - depth (int): depth of decoder layer.
        """
        super(ThreeUp, self).__init__()
        self.channel = 64
        self.depth = depth
        self.total = 5
        self.conv_list = []
        self.list = []

        self.conv = nn.Sequential(nn.Conv2d(in_channels=self.channel * 5, out_channels=self.channel * 5, kernel_size=3, padding="same"), nn.BatchNorm2d(self.channel * 5), nn.ReLU(inplace=True))
        
        for conv in range(1, self.total + 1):
            self.conv_list.append(nn.Conv2d(in_channels=self.channel * (2 ** (conv - 1)), out_channels=self.channel, kernel_size=3, padding="same"))

        for counter in range(1, self.total + 1):
            if self.depth > counter:
                self.list.append(nn.Sequential(nn.MaxPool2d(2 ** (self.depth - counter)), self.conv_list[counter - 1]))
            elif self.depth == counter:
                self.list.append(nn.Sequential(self.conv_list[counter - 1]))
            elif counter == self.total:
                self.list.append(nn.Sequential(nn.Upsample(scale_factor=2 ** (counter - self.depth), mode="bilinear"), self.conv_list[counter - 1]))
            else:
                self.list.append(nn.Sequential(nn.Upsample(scale_factor=2 ** (counter - self.depth), mode="bilinear"), nn.Conv2d(in_channels=self.channel * 5, out_channels=self.channel, kernel_size=3, padding="same")))


    def forward(self, *args):
        """Order: x1, x2, x3, x4, x5 ...
        """
        output = []

        for i in range(self.total):
            x = self.list[i](args[i])
            output.append(x)
        
        x = torch.cat(output, dim=1)
        
        return self.conv(x)



class ThreeOut(nn.Module):
    def __init__(self, channel, out_ch):
        """Output Convolution Layer for UNet 3+ Model. \n
           Args:
           - in_ch (int): input channel of the convolution layer.
           - out_ch (int): output channel of the convolution layer.
        """
        super(ThreeOut, self).__init__()

        self.channel = channel
        self.conv = nn.Sequential(nn.Conv2d(in_channels=self.channel * 5, out_channels=out_ch, kernel_size=1), nn.AdaptiveAvgPool2d(160), nn.Sigmoid())

    def forward(self, x):
        return self.conv(x)



"""Expanded UNet"""
class TripleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(TripleConv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding="same"), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                                nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding="same"), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                                nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding="same"), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


#Input Conv
class ExpInConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        """Input Convolution Layer for Basic UNet Model. \n
           Args:
           - in_ch (int): input channel of the first convolution layer.
           - out_ch (int): output channel of the first and second convolution layer and input channel of the second convolution layer.
        """
        super(ExpInConv, self).__init__()
        self.conv = TripleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(x)


#Down Sampling
class ExpDown(nn.Module):
    def __init__(self, in_ch, out_ch):
        """Down Sampling Layer for Basic UNet Model. \n
           Args:
           - in_ch (int): input channel of first convolution layer.
           - out_ch (int): output channel of the first and second convolution layer and input channel of the second convolution layer.
        """
        super(ExpDown, self).__init__()
        self.down = nn.Sequential(nn.MaxPool2d(2, 2), TripleConv(in_ch, out_ch))

    def forward(self, x):
        return self.down(x)


#Up Sampling
class ExpUp(nn.Module):
    def __init__(self, in_ch, out_ch):
        """Up Sampling Layer for Basic UNet Model. \n
           Args:
           - in_ch (int): input channel of first convolution layer.
           - out_ch (int): output channel of the first and second convolution layer and input channel of the second convolution layer.
        """
        super(ExpUp, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels=in_ch, out_channels=in_ch // 2, kernel_size=2, stride=2)
        self.conv = TripleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x2 = F.pad(x2, (-(diffX // 2), -(diffX - diffX // 2),
                        -(diffY // 2), -(diffY - diffY // 2)))

        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)
