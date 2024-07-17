import torch.nn as nn
from .component import *

class UNet(nn.Module):
    def __init__(self, n_classes):
        super(UNet, self).__init__()
        
        self.inc = DoubleConv(3, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=3, padding=1)
        
    def forward(self, x):
        # (batch, 1, 256, 256) 
        # -> (batch, 64, 256, 256) 
        x1 = self.inc(x)
        
        # (batch, 64, 256, 256) 
        # -> (batch, 64, 128, 128)
        # -> (batch, 128, 128, 128) 
        x2 = self.down1(x1)
        
        # (batch, 128, 128, 128) 
        # -> (batch, 128, 64, 64)
        # -> (batch, 256, 64, 64) 
        x3 = self.down2(x2)
        
        # (batch, 256, 64, 64) 
        # -> (batch, 256, 32, 32) 
        # -> (batch, 512, 32, 32) 
        x4 = self.down3(x3)
        
        # (batch, 512, 32, 32) 
        # -> (batch, 512, 16, 16) 
        # -> (batch, 1024, 16, 16) 
        x5 = self.down4(x4)
        
        # (batch, 1024, 16, 16) 
        # -> (batch, 512, 32, 32) 
        # -> concat 
        # -> (batch, 1024, 32, 32) 
        # -> (batch, 512, 32, 32) 
        x = self.up1(x5, x4)
        
        # (batch, 512, 32, 32) 
        # -> (batch, 256, 64, 64) 
        # -> concat 
        # -> (batch, 512, 64, 64) 
        # -> (batch, 256, 64, 64) 
        x = self.up2(x, x3)
        
        # (batch, 256, 64, 64) 
        # -> (batch, 128, 128, 128) 
        # -> concat 
        # -> (batch, 256, 128, 128) 
        # -> (batch, 128, 128, 128) 
        # -> (batch, 128, 128, 128)
        x = self.up3(x, x2)
        
        # (batch, 128, 128, 128) 
        # -> (batch, 64, 256, 256) 
        # -> concat 
        # -> (batch, 128, 256, 256) 
        # -> (batch, 64, 256, 256) 
        # -> (batch, 64, 256, 256)
        x = self.up4(x, x1)
        
        # (batch, 64, 256, 256) 
        # -> (batch, num_classes, 256, 256)
        x = self.outc(x)
        
        return x