import torch
from ..functional import sdfs as f_sdfs
from ..ops import binary_ops as binops
from ..ops import unary_ops as unops

class TorchSDF(torch.nn.Module):

    def __init__(self):
        super(TorchSDF, self).__init__()

    def bbox(self):
        raise NotImplementedError

    def forward(self, query):
        raise NotImplementedError
