import torch
import torch.nn as nn


class LinearModel(nn.Module):
    def __init__(self, input_dim=12):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    def forward(self, x):
        # x (B, T, N, 1)
        x = self.linear(x)
        return x