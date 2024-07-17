import torch
import torch.nn as nn

class ForwardProcess(nn.Module):
    def __init__(self, image):
        super().__init__()
        # DDPM 노이즈 주입
        # (batch, channel, height, width)
        self.image = image
        self.noise = torch.randn_like(image)
        
    def forward(self):
        return self.image + self.noise
    
class DDPM(nn.Module):
    def __init__(self, forward_process: ForwardProcess):
        super().__init__()
        self.forward_process = forward_process
        
    def forward(self):
        return self.forward_process()
    
image = torch.randn(1, 3, 256, 256)
forward_process = ForwardProcess(image)
ddpm = DDPM(forward_process)
print(ddpm())
