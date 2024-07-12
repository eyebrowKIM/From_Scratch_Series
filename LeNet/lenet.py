import torch
import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        
        # (Batch, 1, 32, 32) -> (Batch, 6, 28, 28) -> (Batch, 6, 14, 14)
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, 5), 
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2))
        
        # (Batch, 6, 14, 14) -> (Batch, 16, 10, 10) -> (Batch, 16, 5, 5)
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, 5), 
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2))
        
        # (Batch, 16*5*5) -> (Batch, 120)
        self.layer3 = nn.Linear(16*5*5, 120)
        # (Batch, 120) -> (Batch, 84)
        self.layer4 = nn.Linear(120, 84) 
        # (Batch, 84) -> (Batch, 10)
        self.layer5 = nn.Linear(84, 10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        
        return out