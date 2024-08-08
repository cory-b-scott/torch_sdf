import torch
from ..functional import sdfs as f_sdfs
from ..ops import binary_ops as binops
from ..ops import unary_ops as unops

from .sdf import TorchSDF

class EulerRepairSDF(TorchSDF):
    def __init__(self, sdf, k=15, device='cpu'):
        super(EulerRepairSDF, self).__init__()
        self.device = device
        self.child = sdf
        self.register_module("child", self.child)
        self.k = k

    def forward(self, query):
        #import matplotlib.pyplot as plt
        #import numpy as np
        dists = self.child(query)
        #return dists
        #og_dists = dists.clone()
        #current_level =
        #signs = dists / torch.sqrt(torch.pow(dists,2.0) + 1e-6)
        #for i in range(self.k):
        new_pts = query.clone()
        dists = self.child(new_pts)
        signs = dists / torch.sqrt(torch.pow(dists,2.0) + 1e-12)
        #dists = self.child(new_pts)
        #dist_accumulator = dists.clone()
        #plt.scatter(*new_pts.detach().cpu().numpy().T,c=0.0*np.ones(len(new_pts)),vmin=0,vmax=self.k+1)
        for i in range(self.k):
            #print(new_pts[:3])
            grads = torch.autograd.grad(dists.sum(), new_pts, retain_graph=True)[0]
            new_pts = new_pts - .125*dists.unsqueeze(-1)*grads
            #print(new_pts[:3])
            #plt.scatter(*new_pts.detach().cpu().numpy().T,c=(i+1.0+np.random.random())*np.ones(len(new_pts)),vmin=0,vmax=self.k+1)
            dists = self.child(new_pts)
            #dist_accumulator += dists
            #new_grads = torch.autograd.grad(new_dists.sum(), dists, retain_graph=True)[0]
            #dists = dists + 3e-3*new_grads
        #    grad_norm = torch.linalg.norm(grads,dim=1,ord=2)
        #    #print(grad_norm.std())
        #    dists = dists - 1e-3*signs*(grad_norm-1)
        #    print(dists,signs*(grad_norm-1))
        #for i in range(self.k):
        #plt.show()
        #print(query[:3])
        #print(torch.linalg.norm(query - new_pts,axis=1))
        dists = torch.linalg.norm(query - new_pts,axis=1)
        dists = dists*signs
        return dists#torch.minimum(dists, og_dists)

    def bbox(self):
        return self.child.bbox()

class DiffusionRepairSDF(TorchSDF):
    def __init__(self, sdf, k=1, device='cpu'):
        super(DiffusionRepairSDF, self).__init__()
        self.device = device
        self.child = sdf
        self.register_module("child", self.child)
        self.k = k
        self.sigma = .1

    def forward(self,query):
        dists = self.child(query)
        normals = torch.autograd.grad(dists.sum(), query, retain_graph=True)[0]
        sampled_normals = normals
        signs = torch.sign(dists)
        for i in range(self.k):
            offset = self.sigma*torch.randn(query.shape)
            new_pts = offset + query
            new_dists = self.child(new_pts)
            new_normals = torch.autograd.grad(new_dists.sum(), new_pts, retain_graph=True)[0]
            sampled_normals = sampled_normals + new_normals

        sampled_normals = sampled_normals / (self.k + 1)
        sampled_normals /= torch.linalg.norm(sampled_normals,dim=1)

        for i in range(3):
            pass
            dists = dists - 1e-3*signs*(sampled_normals-1)
        return dists

    def bbox(self):
        return self.child.bbox()

class ContourRepairSDF(TorchSDF):

    def __init__(self, sdf, device='cpu'):
        super(ContourRepairSDF, self).__init__()
        self.device = device
        self.child = sdf
        self.register_module("child", self.child)

    def forward(self, query):
        dists = self.child(query)
        signs = torch.sign(dists)

        level_set = query[torch.abs(dists) < 1e-3]

        query_np = query.detach().cpu().numpy()
        level_np = level_set.detach().cpu().numpy()
        kdt = KDTree(level_np)
        _, ind = kdt.query(query_np, k=1)
        nearest = level_set[ind][:,0,:]
        #print(query.shape, nearest.shape)
        mod_dists = signs * torch.linalg.norm(query - nearest, axis=1)
        new_dists = torch.minimum(mod_dists, dists)
        return mod_dists

    def bbox(self):
        return self.child.bbox()
