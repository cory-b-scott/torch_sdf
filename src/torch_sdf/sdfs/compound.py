import torch
from ..functional import sdfs as f_sdfs
from ..ops import binary_ops as binops
from ..ops import unary_ops as unops

from .sdf import TorchSDF

class UnionSDF(TorchSDF):

    def __init__(self, sdfs, device='cpu', method='sharp'):
        super(UnionSDF, self).__init__()
        self.sdfs = torch.nn.ModuleList(sdfs)
        self.device = device
        self.method = method

    def forward(self, query):
        dists = torch.stack([sd(query) for sd in self.sdfs])
        if self.method == 'sharp':
            return binops.nary_sharp_union(dists)
        elif self.method == 'smooth_exp':
            return binops.nary_smooth_union_exp(dists, k=1.0/96.0)

    def bbox(self):
        child_bboxes = [item.bbox() for item in self.sdfs]
        lowers = torch.stack([item[0] for item in child_bboxes])
        uppers = torch.stack([item[1] for item in child_bboxes])
        all_pts = torch.cat([lowers,uppers])
        return (all_pts.min(0)[0], all_pts.max(0)[0])

class IntersectionSDF(TorchSDF):

    def __init__(self, sdfs, device='cpu'):
        super(IntersectionSDF, self).__init__()
        self.sdfs = torch.nn.ModuleList(sdfs)
        self.device = device

    def forward(self, query):
        dists = torch.stack([sd(query) for sd in self.sdfs])
        return binops.nary_sharp_intersection(dists)

    def bbox(self):
        child_bboxes = [item.bbox() for item in self.sdfs]
        lowers = torch.stack([item[0] for item in child_bboxes])
        uppers = torch.stack([item[1] for item in child_bboxes])
        all_pts = torch.cat([lowers,uppers])
        return (lowers.max(0)[0], uppers.min(0)[0])

class BlendSDF(TorchSDF):

    def __init__(self, sdfs, weights, device='cpu', method='sharp'):
        super(BlendSDF, self).__init__()
        self.sdfs = torch.nn.ModuleList(sdfs)
        self.weights = torch.nn.Parameter(weights.to(device))
        self.device = device

    def forward(self, query):
        dists = torch.stack([sd(query) for sd in self.sdfs])
        sfweights = self.weights#torch.nn.functional.softmax(torch.pow(self.weights,3.0), dim=0)
        #sfweights = sfweights - .5*sfweights.max()
        #sfweights = torch.nn.functional.relu(sfweights)
        sfweights = torch.nn.functional.softmax(sfweights, dim=0)
        #print(sfweights)
        #print(sfweights)#, torch.pow(sfweights+1,2.0)-1)
        return binops.weighted_sum(dists, sfweights.unsqueeze(1))

class BarycentricSDF(TorchSDF):

    def __init__(self, sdfA, sdfB, weights, device='cpu', method='sharp'):
        super(BarycentricSDF, self).__init__()
        self.device = device
        self.childA = sdfA
        self.childB = sdfB
        self.register_module("childA", self.childA)
        self.register_module("childB", self.childB)
        self.weights = torch.nn.Parameter(weights.to(device))
        self.device = device

    def forward(self, query):
        distsA = self.childA(query)
        distsB = self.childB(query)
        c0, c1, c2, c3 = torch.nn.functional.softmax(self.weights, dim=0)
        return ((c1 + c2) * distsA) + ((c1 + c3) * distsB) + ((c0 - c1 - c2 - c3) * binops.sharp_intersection(distsA, distsB))

class DifferenceSDF(TorchSDF):

    def __init__(self, sdfA, sdfB, device='cpu'):
        super(DifferenceSDF, self).__init__()
        self.device = device
        self.childA = sdfA
        self.childB = sdfB
        self.register_module("childA", self.childA)
        self.register_module("childB", self.childB)

    def forward(self, query):
        return self.childA(query) - self.childB(query)
