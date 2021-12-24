import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np

class Covariance(nn.Module):
    def __init__(self,
                remove_mean=True,
                conv=False,
        ):
        super(Covariance, self).__init__()
        self.remove_mean = remove_mean
        self.conv = conv

    def _remove_mean(self, x):
        x = x.transpose(-1, -2)
        _mean = F.adaptive_avg_pool2d(x, (1,1))
        x = x - _mean
        x = x.transpose(-1, -2)
        return x

    def remove_mean_(self, x):
        _mean = F.adaptive_avg_pool2d(x, (1,1))
        x = x - _mean
        return x

    def _cov(self, x):
        batchsize, d, N = x.size()
        x = x.transpose(-1, -2)
        y = (1. / N ) * (x.bmm(x.transpose(1, 2)))
        return y
    
    def _cross_cov(self, x1, x2):
        batchsize1, N1, d1 = x1.size()
        batchsize2, N2, d2 = x2.size()
        assert batchsize1 == batchsize2
        assert N1 == N2
        x1 = x1.transpose(-1, -2)
        x2 = x2.transpose(-1, -2)

        y = (1. / N1) * (x1.bmm(x2.transpose(1, 2)))
        return y
    
    def cross_cov(self, x1, x2):
        batchsize1, d1, h1, w1 = x1.size()
        batchsize2, d2, h2, w2 = x2.size()
        N1 = h1*w1
        N2 = h2*w2
        assert batchsize1 == batchsize2
        assert N1 == N2
        x1 = x1.view(batchsize1, d1, N1)
        x2 = x2.view(batchsize2, d2, N2)

        y = (1. / N1) * (x1.bmm(x2.transpose(1, 2)))
        return y

    def forward(self, x, y=None):
        if self.remove_mean:
            x = self.remove_mean_(x) if self.conv else self._remove_mean(x)
            if y is not None:
                y = self.remove_mean_(y) if self.conv else self._remove_mean(y)          
        if y is not None:
            if self.conv:
                x = self.cross_cov(x, y)
            else:
                x = self._cross_cov(x, y)
        else:
            x = self._cov(x)
        return x