import torch
import torch.nn as nn

class ePN(nn.Module):
    def __init__(self, eps=1e-8):
        super(ePN, self).__init__()
        self.eps = eps
    def forward(self, x):
        x = torch.sign(x) * torch.sqrt(torch.abs(x) + self.eps)
        x = torch.nn.functional.normalize(x, dim=1)
        return x