import torch
import torch.nn as nn
import torch.nn.functional as F




""" BASIC UNET BLOCKS"""
#Double Conv
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3 ,padding="same"), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                        nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding="same"), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


#Input Conv
class InConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(x)


#Down Sampling
class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down = nn.Sequential(nn.MaxPool2d(2, 2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.down(x)


#Up Sampling
class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
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
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)





""" INCEPTION UNET BLOCKS"""
#Inception Block
class InceptionBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
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
        super().__init__()
        self.conv = nn.Sequential(InceptionBlock(in_ch, out_ch // 4), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                                InceptionBlock(out_ch, out_ch // 4), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


#Input Convolution
class IncInConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = IncDoubleConv(in_ch, out_ch)
    
    def forward(self, x):
        return self.conv(x)


#Down Sampling
class IncDown(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down = nn.Sequential(nn.MaxPool2d(2, 2), IncDoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.down(x)


#Up Sampling
class IncUp(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
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
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1)
        
    def forward(self, x):
        return self.conv(x)




""" ARROW UNET BLOCKS """
#Arrow Block
class ArrowBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        channels = out_ch // 7

        self.x1 = nn.Sequential(nn.Conv2d(in_channels=in_ch, out_channels=channels, kernel_size=3, padding="same"), nn.BatchNorm2d(channels), nn.ReLU(inplace=True))
        self.x2 = nn.Sequential(nn.Conv2d(in_channels=in_ch, out_channels=channels, kernel_size=5, padding="same"), nn.BatchNorm2d(channels), nn.ReLU(inplace=True))
        self.x3 = nn.Sequential(nn.Conv2d(in_channels=in_ch, out_channels=channels, kernel_size=7, padding="same"), nn.BatchNorm2d(channels), nn.ReLU(inplace=True))

        self.x12 = nn.Sequential(nn.Conv2d(in_channels=channels * 2, out_channels=channels, kernel_size=7, padding="same"), nn.BatchNorm2d(channels), nn.ReLU(inplace=True))
        self.x13 = nn.Sequential(nn.Conv2d(in_channels=channels * 2, out_channels=channels, kernel_size=5, padding="same"), nn.BatchNorm2d(channels), nn.ReLU(inplace=True))
        self.x23 = nn.Sequential(nn.Conv2d(in_channels=channels * 2, out_channels=channels, kernel_size=3, padding="same"), nn.BatchNorm2d(channels), nn.ReLU(inplace=True))

        self.x123 = nn.Sequential(nn.Conv2d(in_channels=channels * 3, out_channels=channels, kernel_size=5, padding="same"), nn.BatchNorm2d(channels), nn.ReLU(inplace=True))

    def forward(self, x):
        x1 = self.x1(x)
        x2 = self.x2(x)
        x3 = self.x3(x)

        x12 = self.x12(torch.cat([x1, x2], dim=1))
        x13 = self.x13(torch.cat([x1, x3], dim=1))
        x23 = self.x23(torch.cat([x2, x3], dim=1))

        x123 = self.x123(torch.cat([x12, x13, x23], dim=1))

        return torch.cat([x1, x2, x3, x12, x13, x23, x123], dim=1)


#Input Convolution
class ArrInConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ArrowBlock(in_ch, out_ch)

    def forward(self, x):
        return self.conv(x)


#Down Sampling
class ArrDown(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down = nn.Sequential(nn.MaxPool2d(2, 2), ArrowBlock(in_ch, out_ch))

    def forward(self, x):
        return self.down(x)


#Up Sampling
class ArrUp(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels=in_ch, out_channels=in_ch // 2, kernel_size=2, stride=2)
        self.conv = ArrowBlock(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffX = x2.size()[3] - x1.size()[3]
        diffY = x2.size()[2] - x1.size()[2]

        x2 = F.pad(x2, (-(diffX // 2), -(diffX - diffX // 2),
                        -(diffY // 2), -(diffY - diffY // 2)))
        
        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)


#Out Convolution
class ArrOutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)