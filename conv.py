import numpy as np
import torch
import torch.nn as nn

class Conv(nn.Module):
    def __init__(self):
        super(Conv, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Linear(7, 500),
            nn.ReLU(),
            nn.Linear(500, 10)
            )

    def forward(self, x):
        # x = self.flatten(x)
        logits = self.linear_conv(x)
        return logits

