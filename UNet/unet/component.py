import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )
        
    def forward(self, x):
            
        return self.layer(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        
        # (batch, in_channels, height, width) 
        # -> (batch, in_channels, height/2, width/2)
        # -> (batch, out_channels, height/2-2, width/2-2) 
        # -> (batch, out_channels, height/2-4, width/2-4) 
        
        self.layer = nn.Sequential(
            nn.MaxPool2d(2, 2),
            DoubleConv(in_channels, out_channels)
        )
        
    def forward(self, x):
        
        return self.layer(x)
    
class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        
        # (batch, in_channels, height, width)  
        # -> (batch, in_channels//2, height*2, width*2)
        # -> (batch, out_channels, height*2, width*2-)
        
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
        
    def forward(self, x1, x2):
        
        x1 = self.up(x1)
        
        # 높이와 너비의 차이를 계산
        h_diff = x2.size()[2] - x1.size()[2]
        w_diff = x2.size()[3] - x1.size()[3]    
        
        x2 = x2[:, :, h_diff//2:h_diff//2 + x1.size()[2], w_diff//2:w_diff//2 + x1.size()[3]]
        
        x = torch.cat([x2, x1], dim=1)
        
        x = self.conv(x)
        
        return x
        