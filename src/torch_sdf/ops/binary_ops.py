import torch

def smooth_min_cubic(a, b, k):
    k = k * 6.0
    h = torch.nn.functional.relu( k-torch.abs(a-b))/k
    return torch.minimum(a,b) - h*h*h*k*(1.0/6.0)

def smooth_min_exp(a, b, k):
    r = torch.exp(-a/k) + torch.exp(-b/k);
    return -k*torch.log(r);

def nary_smooth_union_exp(dists, k):
    #print(dists[:10,:])
    #print(torch.exp(-dists / k))
    r = torch.exp(-dists/k)
    #print("^^^^", r[:, :10])
    r = r.sum(0)
    #print(r.max())
    result = -k*torch.log(r)
    result[r==0] = dists.max(0)[0][r==0]
    return result

def sharp_union(a, b):
    return torch.minimum(a,b)

def nary_sharp_union(dists, dim=0):
    return dists.min(dim)[0]

def weighted_sum(dists, weights):
    #print(dists.shape, weights.shape)
    return (dists * weights).sum(0)

def sharp_intersection(a,b):
    return torch.maximum(a,b)

def nary_sharp_intersection(dists, dim=0):
    #print(dists[:10,:])
    return dists.max(dim)[0]
