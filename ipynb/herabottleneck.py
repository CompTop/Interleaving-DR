# This script is not necessary and equivalent to 
# use `from torch_tda.nn import BottleneckLayerHera`
import torch
import torch.nn as nn
from torch.autograd import Function
from hera_tda.bottleneck import BottleneckDistance
import numpy as np
from torch_tda.nn import remove_zero_bars

class BottleneckDistanceHera(Function):
    """
    Compute bottleneck distance between two persistence diagrams

    forward inputs:
        dgm0 - N x 2 torch.float tensor of birth-death pairs
        dgm1 - M x 2 torch.float tensor of birth-death pairs
    """
    @staticmethod
    def forward(ctx, dgm0, dgm1):
        ctx.dtype = dgm0.dtype
        d0 = dgm0.detach().numpy()
        d1 = dgm1.detach().numpy()
        n0 = len(dgm0)
        ctx.n0 = n0
        n1 = len(dgm1)
        ctx.n1 = n1

        dist, match = BottleneckDistance(d0, d1)
        i0, i1 = match

        # TODO check for -1 as index

        ctx.i0 = i0
        ctx.i1 = i1

        d01 = torch.tensor(d0[i0] - d1[i1], dtype=ctx.dtype)
        ctx.d01 = d01
        dist01 = np.linalg.norm(d0[i0] - d1[i1], np.inf)
        ctx.indmax = np.argmax(np.abs(d0[i0] - d1[i1]))

        return torch.tensor(dist01, dtype=ctx.dtype)

    @staticmethod
    def backward(ctx, grad_dist):
        n0 = ctx.n0
        n1 = ctx.n1
        i0 = ctx.i0
        i1 = ctx.i1
        d01 = ctx.d01

        gd0 = torch.zeros(n0, 2, dtype=ctx.dtype)
        gd1 = torch.zeros(n1, 2, dtype=ctx.dtype)


        gd0[i0, ctx.indmax] = np.sign(d01[ctx.indmax]) * grad_dist
        gd1[i1, ctx.indmax] = -np.sign(d01[ctx.indmax]) * grad_dist

        return gd0, gd1


class BottleneckLayerHera(nn.Module):
    def __init__(self):
        super(BottleneckLayerHera, self).__init__()
        self.D = BottleneckDistanceHera()

    def forward(self, dgm0, dgm1):
        dgm0 = remove_zero_bars(dgm0)
        dgm1 = remove_zero_bars(dgm1)

        return self.D.apply(dgm0, dgm1)
