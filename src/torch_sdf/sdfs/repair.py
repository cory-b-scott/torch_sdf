import torch
from ..functional import sdfs as f_sdfs
from ..ops import binary_ops as binops
from ..ops import unary_ops as unops

from .sdf import TorchSDF

class EulerRepairSDF(torch.nn.Module):
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

class DiffusionRepairSDF(torch.nn.Module):
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



class ContourRepairSDF(torch.nn.Module):
    def __init__(self, sdf, k=1, pc = 800, device='cpu'):
        super(ContourRepairSDF, self).__init__()
        self.device = device
        self.child = sdf
        self.register_module("child", self.child)
        self.k = k
        self.pc = pc

    def forward(self, query):
        dists = self.child(query)
        _, idxs = torch.topk(torch.abs(dists), self.pc, largest=False)
        reps = query[idxs, :]
        new_dists = torch.cdist(query, reps).min(1)[0]
        dists = torch.minimum(new_dists * torch.sign(dists), dists)
        return dists

    def bbox(self):
        return self.child.bbox()
