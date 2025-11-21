import torch.nn as nn

class OCRModel(nn.Module):
    def __init__(self):
        super(OCRModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(64*64*64, 256)
        self.classifier = nn.Linear(256, 128)  # modify as per your training config

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.classifier(x)
        return x
