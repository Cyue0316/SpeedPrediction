import torch
import torch.nn as nn

# MLP model
class MLPModel(nn.Module):
    def __init__(self, hidden_dim=64):
        super(MLPModel, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) 
        )

    def forward(self, x):
        # x 的维度是 (B, T, N, 1)
        x = self.mlp(x)
        return x



class LinearModel(nn.Module):
    def __init__(self, input_dim=1):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    def forward(self, x):
        # x (B, T, N, 1)
        x = self.linear(x)
        return x