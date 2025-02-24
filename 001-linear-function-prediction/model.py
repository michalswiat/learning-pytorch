import torch.nn as nn
import torch

class LinearModel(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True).to(device)
        self.bias = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x + self.bias

class LinearModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Linear(in_features=1, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)
