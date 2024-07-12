import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from lenet import LeNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]
)

batch_size = 100

train_loader = datasets.MNIST(root='dataset/',
                                train = True,
                                download = True,
                                transform = transform
                                )

test_loader = datasets.MNIST(root='dataset/',
                                train = False,
                                download = True,
                                transform = transform
                                )

train_data = DataLoader(dataset=train_loader, batch_size=batch_size, shuffle=True)
test_data = DataLoader(dataset=test_loader, batch_size=batch_size, shuffle=True)

model = LeNet().to(device)

def train_lenet():

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
            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = running_corrects / len(train_loader)

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')

    print('Finished Training')
    
def test_lenet():
    with torch.no_grad():
        criterion = nn.CrossEntropyLoss().to(device)

        model.eval()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in test_data:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(test_loader)
        epoch_acc = running_corrects / len(test_loader)
        print('Evaluation')
        print(f'Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')

def main():
    train_lenet()
    test_lenet()
    
if __name__ == '__main__':
    main()