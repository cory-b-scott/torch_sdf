import torch
from ..functional import sdfs as f_sdfs
from ..ops import binary_ops as binops
from ..ops import unary_ops as unops

from .sdf import TorchSDF


class TranslatedSDF(TorchSDF):

    def __init__(self, offset, other, device='cpu'):
        super(TranslatedSDF, self).__init__()
        self.offset = offset
        self.child = other
        self.register_module("child", self.child)
        self.device = device
        try:
            self.register_parameter(name="offset", param=self.offset)
        except:
            pass

    def forward(self, query):
        query_translated = query - self.offset
        return self.child(query_translated)

    def bbox(self):
        child_bbox = self.child.bbox()
        return (self.offset+child_bbox[0], self.offset+child_bbox[1])

class ScaledSDF(TorchSDF):

    def __init__(self, scale, other, device='cpu'):
        super(ScaledSDF, self).__init__()
        self.scale=scale
        self.child = other
        self.register_module("child", self.child)
        self.device = device
        try:
            self.register_parameter(name="scale", param=self.scale)
        except:
            pass

    def forward(self,query):
        scale2 = torch.clamp(self.scale, min=1e-8)
        return scale2*self.child((1.0/scale2)*query)

    def bbox(self):
        child_bbox = self.child.bbox()
        return (self.scale*child_bbox[0], self.scale*child_bbox[1])

class RotatedSDF(TorchSDF):

    def __init__(self, theta, u, v, other, device='cpu'):
        super(RotatedSDF, self).__init__()
        self.theta = theta
        self.u = u
        self.v = v
        self.child = other
        self.register_module("child", self.child)
        self.device = device
        try:
            self.register_parameter(name="theta", param=self.theta)
        except:
            pass
        try:
            self.register_parameter(name="u", param=self.u)
        except:
            pass
        try:
            self.register_parameter(name="v", param=self.v)
        except:
            pass

    def forward(self, query):
        up = self.u / (1e-8 + torch.linalg.norm(self.u))
        vp = self.v / (1e-8 + torch.linalg.norm(self.v))

        rotmat = torch.eye(self.u.shape[0], device=self.u.device, dtype=self.u.dtype)
        rotmat += torch.sin(-1.0*self.theta)*( torch.matmul(vp, up.T) - torch.matmul(up, vp.T)  )
        rotmat += (torch.cos(-1.0*self.theta)-1.0)*( torch.matmul(up, up.T) + torch.matmul(vp, vp.T)  )
        query_rotated = torch.matmul(query.float(), rotmat.T)
        return self.child(query_rotated)
