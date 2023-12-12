import torch.nn as nn
from torch import flatten

class WideLeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, padding=0, stride=1, kernel_size=5)
        self.max1 = nn.MaxPool2d(padding=0, stride=2, kernel_size=2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(20, 50, padding=0, stride=1, kernel_size=5)
        self.max2 = nn.MaxPool2d(padding=0, stride=2, kernel_size=2)
        # 4 * 4 * 50 = 800
        self.fc1 = nn.Linear(800, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        # cnn layer 1
        x = self.conv1(x)
        x = self.max1(x)
        x = self.relu(x)

        # cnn layer 2
        x = self.conv2(x)
        x = self.max2(x)
        x = self.relu(x)

        x = flatten(x, 1)

        # fc layer 1
        x = self.fc1(x)
        x = self.relu(x)

        # fc layer 2
        x = self.fc2(x)

        return x
