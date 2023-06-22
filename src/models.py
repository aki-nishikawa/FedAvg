import torch.nn as nn


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.base = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 128, 3),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        h = self.base(x)
        h_size = h.size()
        x = self.base(x).reshape(h_size[0], -1, h_size[2] ** 2).mean(2)
        x = self.head(x)
        return x
