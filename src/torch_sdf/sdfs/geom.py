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

class QuadraticBezierSDF(TorchSDF):

    def __init__(self, A, B, C, device='cpu'):
        super(QuadraticBezierSDF, self).__init__()

        self.A = A
        self.B = B
        self.C = C
        self.device = device
        try:
            self.register_parameter(name="bounds", param=self.bounds)
        except:
            pass

    def forward(self,query):
        return f_sdfs.quad_bezier_sdf(query, self.A, self.B, self.C)

class LineSegmentSDF(TorchSDF):

    def __init__(self, A, B, device='cpu'):
        super(LineSegmentSDF, self).__init__()

        self.A = A
        self.B = B
        self.device = device

    def forward(self, query):
        return f_sdfs.line_seg_2d_sdf(query, self.A, self.B)

class PolyLineSDF(TorchSDF):

    def __init__(self, ptslist, device='cpu'):
        super(PolyLineSDF, self).__init__()
        self.lines = torch.nn.ModuleList([LineSegmentSDF(pA, pB, device=device) for pA,pB in zip(ptslist[:-1], ptslist[1:])])

        self.device = device

    def forward(self, query):
        dists = torch.stack([line(query) for line in self.lines])
        signs = torch.sign(dists)
        dists = torch.abs(dists)
        return signs.max(0)[0]*dists.min(0)[0]

class RecursiveBezierSDF(TorchSDF):

    def __init__(self, ptslist, device='cpu'):
        super(RecursiveBezierSDF, self).__init__()

        if len(ptslist) == 3:
            self.childA = LineSegmentSDF(ptslist[0], ptslist[1],device=device)
            self.childB = LineSegmentSDF(ptslist[1], ptslist[2],device=device)
        else:
            self.childA = RecursiveBezierSDF(ptslist[:-1], device=device)
            self.childB = RecursiveBezierSDF(ptslist[1:], device=device)

        self.device = device

    def forward(self, query):
        qA = self.childA(query)
        qB = self.childB(query)

        sA = -1*torch.sign(qA)
        sB = -1*torch.sign(qB)

        gradA = torch.autograd.grad(qA.sum(), query, retain_graph=True)[0]
        gradB = torch.autograd.grad(qB.sum(), query, retain_graph=True)[0]

        nearestA = query + (qA.unsqueeze(1) * gradA)
        nearestB = query + (qB.unsqueeze(1) * gradB)


        tvals = torch.linspace(0,1,32, device=self.device).unsqueeze(-1).unsqueeze(-1)

        #print(nearestA.device, nearestB.device, tvals.device)
        #print(query.shape, nearestA.shape, nearestB.shape, tvals.shape)
        qp = (1-tvals)*nearestA + (tvals)*nearestB

        dists = torch.linalg.norm(query.unsqueeze(0) - qp, axis=-1)
        #print(dists.shape)
        dists = dists.min(0)[0]
        #dists = torch.cdist(query, qp).min(1)[0]

        return torch.maximum(sA, sB)*dists
