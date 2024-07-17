import os
import subprocess
import time
from datetime import datetime
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from torchvision import transforms

from unet import UNet

class Trainer:
    def __init__(self, train_data, num_epochs, lr, gpu_id, save_model, tensorboard):
        self.train_data = train_data
        self.num_epochs = num_epochs
        self.lr = lr
        self.gpu_id = gpu_id
        self.save_model = save_model
        self.eval = eval
        self.tensorboard = tensorboard  
        self.set_device()
        self.start_time = str(time.time())
    
    def set_device(self):
        self.device = torch.device(f'cuda:{self.gpu_id}' if torch.cuda.is_available() else 'cpu')
    
    def train_unet(self):
        device = self.device
        model = UNet(n_classes=1000).to(device)
        
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        num_epochs = 10

        for epoch in tqdm(range(num_epochs)):
            model.train()
            running_loss = 0.0

            for inputs, labels in tqdm(self.train_data, total=len(self.train_data), leave=False):
                optimizer.zero_grad()
                inputs = inputs.to(device)
                labels = labels.to(device)
                labels = labels.squeeze(1)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            epoch_loss = running_loss / len(self.train_data)
            
            if self.tensorboard:
                current_time = int(time.time())
                self.log_dir = './runs/run_{}'.format(self.start_time)
                
                writer = SummaryWriter(log_dir=self.log_dir + './' + str(current_time))
                writer.add_scalar('Loss/train', epoch_loss, epoch)
                writer.close()

            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

        print('Finished Training')
        
        if self.save_model:
            save_time = datetime.now().strftime('%Y%m%d%H%M%S')
            if 'result' not in os.listdir():
                os.mkdir('./result')
            torch.save(model.state_dict(), './result/unet_{}.pth'.format(save_time))
            
        self.model = model
        
        
    def launch_tensorboard(self):
        subprocess.run(['tensorboard', '--logdir', self.log_dir])