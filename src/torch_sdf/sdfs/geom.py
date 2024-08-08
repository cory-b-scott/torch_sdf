import torch
from ..functional import sdfs as f_sdfs
from ..ops import binary_ops as binops
from ..ops import unary_ops as unops

from .sdf import TorchSDF

class SphereSDF(TorchSDF):

    def __init__(self, rad, device='cpu'):
        super(SphereSDF, self).__init__()
        self.rad = rad
        self.device = device
        try:
            self.register_parameter(name="rad", param=self.rad)
        except:
            pass

    def forward(self,query):
        rad2 = self.rad
        return f_sdfs.sphere_sdf(query, rad2)

    def bbox(self):
        return (-1*self.rad, self.rad)

class Triangle2DSDF(TorchSDF):

    def __init__(self, pts, device='cpu'):
        super(Triangle2DSDF, self).__init__()

        self.p0 = pts[0]
        self.p1 = pts[1]
        self.p2 = pts[2]
        self.device = device
        try:
            self.register_parameter(name="p0", param=self.p0)
        except:
            pass
        try:
            self.register_parameter(name="p1", param=self.p1)
        except:
            pass
        try:
            self.register_parameter(name="p2", param=self.p2)
        except:
            pass

    def forward(self,query):
        return f_sdfs.triangle_2D(query, self.p0+SMALL_POS_NUM, self.p1-SMALL_POS_NUM, self.p2)

class RectSDF(TorchSDF):

    def __init__(self, bounds, device='cpu'):
        super(RectSDF, self).__init__()

        self.bounds = bounds
        self.device = device
        try:
            self.register_parameter(name="bounds", param=self.bounds)
        except:
            pass

    def forward(self,query):
        return f_sdfs.axis_aligned_rect_sdf(query, self.bounds)
