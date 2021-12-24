import torch
import torch.nn as nn

class scaleNorm(nn.Module):
    def __init__(self, factor, is_learnable=False):
        super(scaleNorm, self).__init__()
        if is_learnable:
            self.factor = nn.Parameter(torch.tensor(factor))
        else:
            self.factor = factor
    def forward(self, x):
        # import pdb;pdb.set_trace()
        return self.factor * x

