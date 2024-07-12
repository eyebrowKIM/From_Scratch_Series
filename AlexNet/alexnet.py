import torch.nn as nn

class AlexNet(nn.Module):
    # TODO : 채널 반씩 나눠서 2개의 GPU에 병렬로 처리하도록 구현
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()
        # (Batch, 3, 224, 224) -> (Batch, 96, 54, 54)
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        # (Batch, 96, 54, 54) -> (Batch, 256, 27, 27)
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        # (Batch, 256, 27, 27) -> (Batch, 384, 13, 13)
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        # (Batch, 384, 13, 13) -> (Batch, 384, 13, 13)
        self.layer4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        # (Batch, 384, 13, 13) -> (Batch, 256, 13, 13)
        self.layer5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        # (Batch, 256*13*13) -> (Batch, 4096) -> (Batch, 4096) -> (Batch, num_classes)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 13 * 13, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )
        
    def forward(self, x):
        assert x.shape[1:] == (3, 224, 224), f"Input shape should be (3, 224, 224), but got {x.shape[1:]}"
        
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        
        return out