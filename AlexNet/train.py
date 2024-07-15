import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from alexnet import AlexNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 100

def train_alexnet(train_data):

    model = AlexNet(num_classes=1000).to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_data:
            optimizer.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        
        if len(train_loader.dataset) % batch_size == 0:
            epoch_acc = running_corrects / len(train_loader.dataset)
            
def main():
    train_alexnet()
    
if __name__ == '__main__':
    main()