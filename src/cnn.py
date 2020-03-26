import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim


class CNN(nn.Module):
    def __init__(self, alpha, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5)
        self.lin = nn.Linear(20 * 20 * 32, 10)

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr = alpha)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.to(self.device)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.lin(x.view(-1, 20 * 20 * 32))
        x = F.log_softmax(x, dim = 1)
        return x




