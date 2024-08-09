import torch

def dot2(v):
    return dot(v,v)

def dot(v1, v2):
    #print(v1.shape, v2.shape)
    tr = (v1*v2)
    #print("$$$",tr.shape)
    return tr.sum(1)

def rev_dot_2D(v1, v2):
    #print(v1.shape, v2.shape)
    return v1[:,0]*v2[:,1] - v1[:,1]*v2[:,0]

def sphere_sdf(query, rad):
    #return torch.sqrt(torch.pow(torch.real(query),2.0).sum(-1)) - rad + 1j*torch.imag(torch.sqrt(torch.pow(query,2.0).sum(-1)))
    return torch.linalg.norm(query, axis=1) - rad

def axis_aligned_rect_sdf(query, bounds):
    q = torch.abs(query) - bounds
    return torch.linalg.norm(torch.nn.functional.relu(q),axis=1) - torch.nn.functional.relu(-q.max(1)[0])

def triangle_2D(query, p0, p1, p2):
    e0 = (p1-p0).unsqueeze(0)
    e1 = (p2-p1).unsqueeze(0)
    e2 = (p0-p2).unsqueeze(0)
    v0 = query - p0
    v1 = query - p1
    v2 = query - p2
    #print("%%%",e0*torch.clamp( dot(v0,e0)/dot(e0,e0), 0.0, 1.0 ).unsqueeze(1))
    pq0 = v0 - e0*torch.clamp( dot(v0,e0)/dot(e0,e0), 0.0, 1.0 ).unsqueeze(1)
    pq1 = v1 - e1*torch.clamp( dot(v1,e1)/dot(e1,e1), 0.0, 1.0 ).unsqueeze(1)
    pq2 = v2 - e2*torch.clamp( dot(v2,e2)/dot(e2,e2), 0.0, 1.0 ).unsqueeze(1)
    #print(pq0)
    #quit()
    s = torch.sign( rev_dot_2D(e0, e2)  )
    dX = torch.stack([dot(pq0, pq0), dot(pq1, pq1), dot(pq2, pq2)])
    dY = s*torch.stack([rev_dot_2D(v0, e0), rev_dot_2D(v1, e1), rev_dot_2D(v2,e2)])

    #vec2(dot(pq0,pq0), s*(v0.x*e0.y-v0.y*e0.x)),
    #                 vec2(dot(pq1,pq1), s*(v1.x*e1.y-v1.y*e1.x))),
    #                 vec2(dot(pq2,pq2), s*(v2.x*e2.y-v2.y*e2.x)));
    return -torch.sqrt(dX.min(0)[0])*torch.sign(dY.min(0)[0]);

def quad_bez_case1(sel, h, q, kx, b,c,d):
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print(h)
    h = torch.nan_to_num(torch.sqrt(sel*h), nan=0)
    print(h)
    x = (torch.stack([h,-h])-q)/2.0
    print(x)
    uv = torch.sign(x)*torch.pow(torch.abs(x), 1.0/3.0)
    print(uv)
    t = torch.clamp( uv.sum(1)-kx, min=0.0, max=1.0 )
    print(t)
    res = dot2(d + (c + b*t)*t)
    print("%%",res)
    return sel*torch.sqrt(res)

def quad_bez_case2(sel, p, q, kx, b,c,d):
    print("************************")
    z = torch.sqrt(sel * (-p));
    print(z)
    inner = sel*q/(sel*p*z*2.0)
    inner = torch.nan_to_num(inner,nan=0.0)
    print(inner)
    v = torch.acos( inner ) / 3.0;
    print(v)
    m = torch.cos(v);
    print(m)
    n = torch.sin(v)*1.732050808;
    print(n)
    t = torch.clamp(torch.stack([m+m,-n-m,n-m])*z-kx,0.0,1.0).T
    print(t)
    res = torch.minimum( dot2(d+(c+b*t[:,0:1])*t[:,0:1]),
               dot2(d+(c+b*t[:,1:2])*t[:,1:2]) )
    print(res)
    #quit()
    return sel*torch.sqrt(res)

def quad_bezier_sdf(pos, A, B, C):
    A = A.unsqueeze(0)
    B = B.unsqueeze(0)
    C = C.unsqueeze(0)

    a = B - A
    b = A - 2.0*B + C
    c = a * 2.0
    d = A - pos
    #print([item.shape for item in [a,b,c,d]])
    kk = 1.0/dot(b,b)
    kx = kk * dot(a,b)
    ky = kk * (2.0*dot(a,a)+dot(d,b)) / 3.0
    kz = kk * dot(d,a)

    print([item.shape for item in [kk,kx,ky,kz]])
    #quit()
    p = ky - kx*kx
    p3 = p*p*p
    q = kx*(2.0*kx*kx-3.0*ky) + kz
    h = q*q + 4.0*p3
    print([item.shape for item in [p, p3, q, h]])
    quit()
    sel = torch.sign(torch.nn.functional.relu(h))
    case1_res = quad_bez_case1(sel,   h, q, kx, b,c,d)
    case2_res = quad_bez_case2(1-sel, p, q, kx, b,c,d)

    print("$$$",case1_res, case2_res)

    return case1_res + case2_res

def polygon(query, pts):
    pass

""" TODO:
    - Box
    - Rounded Box
    - Box Frame
    - Torus
    - Capped Torus
    - Link
    - Infinite Cylinder
    - Cone
    - Infinite Cone
    - Hexagonal Prism

"""
