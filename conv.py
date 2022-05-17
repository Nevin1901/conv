import numpy as np
import torch
import torch.nn as nn

class Conv(nn.Module):
        def __init__(self):
            super(Conv, self).__init__()
            self.flatten = nn.Flatten()
            self.linear_conv = nn.Sequential(
                    nn.Linear(28*28, 512),
                    nn.Conv2d(kernel_size=5),
                    nn.ReLU(),
                    nn.AvgPool2d(kernel_size=5),
                    nn.Linear(28*28, 512),
                    nn.ReLU(),
                    nn.Linear(512, 10)
                    )


    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_conv(x)
        return logits

