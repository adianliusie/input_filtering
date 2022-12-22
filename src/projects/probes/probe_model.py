import torch
import torch.nn as nn

class LinearProbe(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(768, 2)
    
    def forward(self, H):
        y = self.linear(H)
        return y
