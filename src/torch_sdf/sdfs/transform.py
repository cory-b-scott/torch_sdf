import torch
from ..functional import sdfs as f_sdfs
from ..ops import binary_ops as binops
from ..ops import unary_ops as unops

from .sdf import TorchSDF


class RoundSDF(TorchSDF):

    def __init__(self, rad, other, device='cpu'):
        super(RoundSDF, self).__init__()
        self.rad = rad
        self.device = device
        self.child = other
        self.register_module("child", self.child)
        try:
            self.register_parameter(name="rad", param=self.rad)
        except:
            pass

    def forward(self, query):
        #rad2 = torch.clamp(self.rad, min=1e-6)
        rad2 = self.rad
        return unops.round( self.child(query),  rad2  )

    def bbox(self):
        cb = self.child.bbox()
        return (cb[0] - self.rad, cb[1] + self.rad)


class InvertSDF(TorchSDF):

    def __init__(self, other, device='cpu'):
        super(InvertSDF, self).__init__()
        self.device = device
        self.child = other

    def forward(self, query):
        return -1*self.child(query)

class ElongateSDF(TorchSDF):

    def __init__(self, rad, other, device='cpu'):
        super(ElongateSDF, self).__init__()
        self.rad = rad
        self.device = device
        self.child = other
        self.register_module("child", self.child)
        try:
            self.register_parameter(name="rad", param=self.rad)
        except:
            pass

    def forward(self, query):
        #q = torch.abs(query) - self.rad
        #return self.child(torch.nn.functional.relu(q)) - torch.nn.functional.relu(-q.max(1)[0])
        rad2 = torch.clamp(self.rad, min=SMALL_POS_NUM)
        q = query - torch.clamp(query, -rad2, rad2)
        return self.child(q)

class OnionSDF(TorchSDF):

    def __init__(self, rad, other, device='cpu'):
        super(OnionSDF, self).__init__()
        self.rad = rad
        self.device = device
        self.child = other
        self.register_module("child", self.child)

    def forward(self, query):
        rad2 = self.rad#torch.clamp(self.rad, min=SMALL_POS_NUM)
        return unops.onion( self.child(query),  rad2  )

    def bbox(self):
        cb = self.child.bbox()
        return (cb[0] - self.rad, cb[1] + self.rad)


class NoisySDF(TorchSDF):
    def __init__(self, rad, other, device='cpu'):
        super(NoisySDF, self).__init__()
        self.rad = rad
        self.device = device
        self.child = other
        self.register_module("child", self.child)
        try:
            self.register_parameter(name="rad", param=self.rad)
        except:
            pass

    def forward(self, query):
        return unops.add_normal_noise(self.child(query), self.rad)

    def bbox(self):
        cb = self.child.bbox()
        return (cb[0] - self.rad, cb[1] + self.rad)

class LocalAverageSDF(TorchSDF):
    def __init__(self, radius, sdf, k=10, device='cpu'):
        super(LocalAverageSDF, self).__init__()
        self.device = device
        self.child = sdf
        self.register_module("child", self.child)
        self.rad = radius
        self.k = k

    def forward(self, query):
        subdists = [
            self.child(query + self.rad*torch.randn(query.shape, device=query.device)) for i in range(self.k)
        ]
        return torch.stack(subdists, axis=-1).mean(-1)

    def bbox(self):
        cb = self.child.bbox()
        return (cb[0] - self.rad, cb[1] + self.rad)
