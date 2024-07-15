import os
import subprocess
import time
from datetime import datetime

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

from lenet import LeNet

class Trainer:
    def __init__(self, train_data, test_data, num_epochs, lr, gpu_id, save_model, eval, tensorboard):
        self.train_data = train_data
        self.test_data = test_data
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
    
    def train_lenet(self):
        device = self.device
        model = LeNet().to(device)
        
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        num_epochs = 10

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in self.train_data:
                optimizer.zero_grad()
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(self.train_data)
            epoch_acc = running_corrects / len(self.train_data)
            
            if self.tensorboard:
                current_time = int(time.time())
                self.log_dir = './runs/run_{}'.format(self.start_time)
                
                writer = SummaryWriter(log_dir=self.log_dir + './' + str(current_time))
                writer.add_scalar('Loss/train', epoch_loss, epoch)
                writer.add_scalar('Accuracy/train', epoch_acc, epoch)
                writer.close()

            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')

        print('Finished Training')
        
        if self.save_model:
            save_time = datetime.now().strftime('%Y%m%d%H%M%S')
            if 'result' not in os.listdir():
                os.mkdir('./result')
            torch.save(model.state_dict(), './result/lenet_{}.pth'.format(save_time))
            
        self.model = model
        
        if self.eval:
            self.test_lenet()
        
    def test_lenet(self):
        with torch.no_grad():
            device = self.device
            
            criterion = nn.CrossEntropyLoss().to(device)

            model = self.model.eval().to(device)
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in self.test_data:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(self.test_data)
            epoch_acc = running_corrects / len(self.test_data)
            print('Evaluation')
            print(f'Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')
            
    def launch_tensorboard(self):
        subprocess.run(['tensorboard', '--logdir', self.log_dir])