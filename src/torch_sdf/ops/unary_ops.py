import torch

def round(a, r):
    return a - r

def onion(a, r):
    return torch.abs(a) - r

def add_normal_noise(a, r):
    return a + r*torch.randn(a.shape, device=a.device)
