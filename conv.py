import numpy as np
import torch
import torch.nn as nn

class Conv(nn.Module):
    def __init__(self):
        super(Conv, self).__init__()
        self.flatten = nn.Flatten()
        self.cnn1 = nn.Conv2d(1, 6, 5)
        self.max_pool1 = nn.MaxPool2d(2, 2)
        self.cnn_layers = nn.Sequential(
                # 1 input since greyscale
                nn.Conv2d(1, 28, 5),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(28, 56, 5),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
            )
        self.linear_layers = nn.Sequential(
                nn.Linear(896, 400),
                nn.ReLU(),
                nn.Linear(400, 84),
                nn.ReLU(),
                nn.Linear(84, 10)
                )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = self.flatten(x)
        x = self.linear_layers(x)
        return x

