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





""" DILATION BLOCKS"""
#Dilation Block
class DilationBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        """Dilation Block for Dilation UNet Model. \n
           Args:
           - in_ch (int): input channel of the convolution layer.
           - out_ch (int): output channel of the convolution layer.
        """
        super(DilationBlock, self).__init__()
        self.x1 = nn.Sequential(nn.Conv2d(in_channels=in_ch, out_channels=out_ch // 4, kernel_size=3, padding="same", dilation=1), nn.BatchNorm2d(out_ch // 4), nn.ReLU(inplace=True))
        self.x2 = nn.Sequential(nn.Conv2d(in_channels=in_ch, out_channels=out_ch // 4, kernel_size=3, padding="same", dilation=2), nn.BatchNorm2d(out_ch // 4), nn.ReLU(inplace=True))
        self.x3 = nn.Sequential(nn.Conv2d(in_channels=in_ch, out_channels=out_ch // 4, kernel_size=3, padding="same", dilation=3), nn.BatchNorm2d(out_ch // 4), nn.ReLU(inplace=True))
        self.x4 = nn.Sequential(nn.Conv2d(in_channels=in_ch, out_channels=out_ch // 4, kernel_size=3, padding="same", dilation=4), nn.BatchNorm2d(out_ch // 4), nn.ReLU(inplace=True))

    def forward(self, x):
        x1 = self.x1(x)
        x2 = self.x2(x)
        x3 = self.x3(x)
        x4 = self.x4(x)
        x = torch.cat([x1, x2, x3, x4], dim=1)
        return x


#Double Conv
class DilDoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        """Double Dilation Convolution for Dilation UNet Model. \n
           Args:
           - in_ch (int): input channel of the first convolution layer.
           - out_ch (int): output channel of the first and second convolution layer and input channel of the second convolution layer.
        """
        super(DilDoubleConv, self).__init__()
        self.conv = nn.Sequential(DilationBlock(in_ch, out_ch), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                                DilationBlock(out_ch, out_ch), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


#Input Convolution
class DilInConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        """Input Convolution Layer for Dilation UNet Model. \n
           Args:
           - in_ch (int): input channel of the convolution layer.
           - out_ch (int): output channel of the first and second convolution layer and input channel of the second convolution layer.
        """
        super(DilInConv, self).__init__()
        self.conv = DilDoubleConv(in_ch, out_ch)
    
    def forward(self, x):
        return self.conv(x)


#Down Sampling
class DilDown(nn.Module):
    def __init__(self, in_ch, out_ch):
        """Down Sampling Layer for Dilation UNet Model. \n
           Args:
           - in_ch (int): input channel of the convolution layer.
           - out_ch (int): output channel of the first and second convolution layer and input channel of the second convolution layer.
        """
        super(DilDown, self).__init__()
        self.down = nn.Sequential(nn.MaxPool2d(2, 2), DilDoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.down(x)


#Up Sampling
class DilUp(nn.Module):
    def __init__(self, in_ch, out_ch):
        """Up Sampling Layer for Dilation UNet Model. \n
           Args:
           - in_ch (int): input channel of the convolution layer.
           - out_ch (int): output channel of the first and second convolution layer and input channel of the second convolution layer.
        """
        super(DilUp, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels=in_ch, out_channels=in_ch // 2, kernel_size=2, stride=2)
        self.conv = DilDoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffX = x2.size()[3] - x1.size()[3]
        diffY = x2.size()[2] - x1.size()[2]

        x2 = F.pad(x2, (-(diffX // 2), -(diffX - diffX // 2),
                        -(diffY // 2), -(diffY - diffY // 2)))
        x = torch.cat([x2, x1], dim=1)
        
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

        self.down_list = nn.ModuleList()
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

        self.up_list = nn.ModuleList()
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


"""DİLATİON+ UNET"""
#Dilation+ Block
class DilationPlusBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DilationPlusBlock, self).__init__()
        self.x1 = nn.Sequential(nn.Conv2d(in_channels=in_ch, out_channels=out_ch // 2, kernel_size=3, padding="same", dilation=1))
        self.x2 = nn.Sequential(nn.Conv2d(in_channels=in_ch, out_channels=out_ch // 2, kernel_size=3, padding="same", dilation=2))


    def forward(self, x):
        x1 = self.x1(x)
        x2 = self.x2(x)
        
        x = torch.cat([x1, x2], dim=1)
        return x


#Dilation+ Double Convolution
class DilPlusDoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DilPlusDoubleConv, self).__init__()
        self.conv1 = nn.Sequential(DilationPlusBlock(in_ch, out_ch // 2), nn.BatchNorm2d(out_ch // 2), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(DilationPlusBlock(out_ch // 2, out_ch // 2), nn.BatchNorm2d(out_ch // 2), nn.ReLU(inplace=True))



    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x = torch.cat([x1, x2], dim=1)

        return x


#Input Convolution
class DilPlusInConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        """Input Convolution Layer for Dilation+ UNet Model. \n
           Args:
           - in_ch (int): input channel of the convolution layer.
           - out_ch (int): output channel of the first and second convolution layer and input channel of the second convolution layer.
        """
        super(DilPlusInConv, self).__init__()
        self.conv = DilPlusDoubleConv(in_ch, out_ch)
    
    def forward(self, x):
        return self.conv(x)


#Down Sampling
class DilPlusDown(nn.Module):
    def __init__(self, in_ch, out_ch):
        """Down Sampling Layer for Dilation+ UNet Model. \n
           Args:
           - in_ch (int): input channel of the convolution layer.
           - out_ch (int): output channel of the first and second convolution layer and input channel of the second convolution layer.
        """
        super(DilPlusDown, self).__init__()
        self.down = nn.Sequential(nn.MaxPool2d(2, 2), DilPlusDoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.down(x)


#Up Sampling
class DilPlusUp(nn.Module):
    def __init__(self, in_ch, out_ch):
        """Up Sampling Layer for Dilation+ UNet Model. \n
           Args:
           - in_ch (int): input channel of the convolution layer.
           - out_ch (int): output channel of the first and second convolution layer and input channel of the second convolution layer.
        """
        super(DilPlusUp, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels=in_ch, out_channels=in_ch // 2, kernel_size=2, stride=2)
        self.conv = DilPlusDoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffX = x2.size()[3] - x1.size()[3]
        diffY = x2.size()[2] - x1.size()[2]

        x2 = F.pad(x2, (-(diffX // 2), -(diffX - diffX // 2),
                        -(diffY // 2), -(diffY - diffY // 2)))
        x = torch.cat([x2, x1], dim=1)
        
        return self.conv(x)



"""Dilation++ UNet"""
#Dilation++ Double Convolution
class Dil2PlusDoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Dil2PlusDoubleConv, self).__init__()
        self.conv1 = nn.Sequential(DilationBlock(in_ch, out_ch // 2), nn.BatchNorm2d(out_ch // 2), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(DilationBlock(out_ch // 2, out_ch // 2), nn.BatchNorm2d(out_ch // 2), nn.ReLU(inplace=True))



    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x = torch.cat([x1, x2], dim=1)

        return x


#Input Convolution
class Dil2PlusInConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        """Input Convolution Layer for Dilation++ UNet Model. \n
           Args:
           - in_ch (int): input channel of the convolution layer.
           - out_ch (int): output channel of the first and second convolution layer and input channel of the second convolution layer.
        """
        super(Dil2PlusInConv, self).__init__()
        self.conv = Dil2PlusDoubleConv(in_ch, out_ch)
    
    def forward(self, x):
        return self.conv(x)


#Down Sampling
class Dil2PlusDown(nn.Module):
    def __init__(self, in_ch, out_ch):
        """Down Sampling Layer for Dilation++ UNet Model. \n
           Args:
           - in_ch (int): input channel of the convolution layer.
           - out_ch (int): output channel of the first and second convolution layer and input channel of the second convolution layer.
        """
        super(Dil2PlusDown, self).__init__()
        self.down = nn.Sequential(nn.MaxPool2d(2, 2), Dil2PlusDoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.down(x)


#Up Sampling
class Dil2PlusUp(nn.Module):
    def __init__(self, in_ch, out_ch):
        """Up Sampling Layer for Dilation++ UNet Model. \n
           Args:
           - in_ch (int): input channel of the convolution layer.
           - out_ch (int): output channel of the first and second convolution layer and input channel of the second convolution layer.
        """
        super(Dil2PlusUp, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels=in_ch, out_channels=in_ch // 2, kernel_size=2, stride=2)
        self.conv = Dil2PlusDoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffX = x2.size()[3] - x1.size()[3]
        diffY = x2.size()[2] - x1.size()[2]

        x2 = F.pad(x2, (-(diffX // 2), -(diffX - diffX // 2),
                        -(diffY // 2), -(diffY - diffY // 2)))
        x = torch.cat([x2, x1], dim=1)
        
        return self.conv(x)