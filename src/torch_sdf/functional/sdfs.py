import torch

def dot2(v):
    return dot(v,v)

def dot(v1, v2):
    tr = (v1*v2)
    return tr.sum(1)

def cross2D(v1, v2):
    return v1[:,0]*v2[:,1] - v1[:,1]*v2[:,0]

def sphere_sdf(query, rad):
    return torch.linalg.norm(query, axis=1) - rad

def axis_aligned_bounding_box_sdf(query, bounds):
    center = (bounds[1] + bounds[0]) / 2.0
    cent_query = query - center
    new_bounds = bounds[1] - center
    return axis_aligned_rect_sdf(cent_query, new_bounds)

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

    pq0 = v0 - e0*torch.clamp( dot(v0,e0)/dot(e0,e0), 0.0, 1.0 ).unsqueeze(1)
    pq1 = v1 - e1*torch.clamp( dot(v1,e1)/dot(e1,e1), 0.0, 1.0 ).unsqueeze(1)
    pq2 = v2 - e2*torch.clamp( dot(v2,e2)/dot(e2,e2), 0.0, 1.0 ).unsqueeze(1)

    s = torch.sign( cross2D(e0, e2)  )

    dX = torch.stack([dot(pq0, pq0), dot(pq1, pq1), dot(pq2, pq2)])

    dY = s*torch.stack([cross2D(v0, e0), cross2D(v1, e1), cross2D(v2,e2)])

    return -torch.sqrt(dX.min(0)[0])*torch.sign(dY.min(0)[0]);

def line_seg_2d_sdf(p, a, b):
    pa = p-a
    ba = (b-a).unsqueeze(0)
    h = torch.clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0).unsqueeze(1)
    #print(ba.shape, h.shape)
    cross = torch.sign(cross2D(pa,ba))

    return cross*torch.linalg.norm( pa - ba*h, axis=1 )

def moon_2d_sdf(p, d, ra, rb):

    py = torch.abs(p[:,1])
    a = (ra*ra - rb*rb + d*d)/(2.0*d)
    b = torch.sqrt(torch.nn.functional.relu(ra*ra-a*a))
    ab = torch.stack([a,b]).unsqueeze(0)
    mask = d*(p[:,0]*b - py*a) > d*d*torch.nn.functional.relu(b-py)
    mask = mask.float()

    p = torch.stack([p[:,0],py],1)

    case1 = torch.linalg.norm(p - ab, axis=1)
    case2 = torch.maximum( torch.linalg.norm(p, axis=1) - ra, -1*(torch.linalg.norm(p - torch.stack([d,torch.zeros_like(d)]).unsqueeze(0), axis=1)-rb)   )

    return case1*mask + (1-mask)*case2

def parabola_sdf_2d(pos, wi, he):
    pos = torch.stack([torch.abs(pos[:,0]), pos[:,1]], 1)
    ik = wi*wi/he
    p = ik*(he-pos[:,1]-0.5*ik)/3.0
    q = pos[:,0]*ik*ik*0.25
    h = q*q - p*p*p
    r = torch.sqrt(torch.abs(h));
    mask = torch.sign(torch.nn.functional.relu(h))

    x = mask * torch.nan_to_num(torch.pow(q+r,1.0/3.0) - torch.pow(torch.abs(q-r),1.0/3.0)*torch.sign(r-q),nan=0) + \
        torch.nan_to_num((1-mask)*(2.0*torch.cos(torch.atan(r/q)/3.0)*torch.sqrt(p)), nan=0)

    x = torch.minimum(x,wi*torch.ones_like(x))

    return torch.linalg.norm(pos-torch.stack([x,he-x*x/ik],1), axis=1) * \
           torch.sign(ik*(pos[:,1]-he)+pos[:,0]*pos[:,0])

def quad_bez_case1(sel, h, p, q, kx, b,c,d):
    h = torch.nan_to_num(torch.sqrt(h), nan=0)
    x = ((torch.stack([h,-h])-q)/2.0).T
    uv = torch.sign(x)*torch.pow(torch.abs(x), 1.0/3.0)
    t= uv.sum(1,keepdim=True)
    p = p.unsqueeze(1)
    q = q.unsqueeze(1)
    t = t - (t*(t*t+3.0*p)+q)/(3.0*t*t+3.0*p)
    t = torch.clamp( t-kx, min=0.0, max=1.0 );
    w = d+(c+b*t)*t;
    res = dot2(w);

    return torch.nan_to_num(sel*torch.sqrt(res), nan=0)

def quad_bez_case2(sel, p, q, kx, b,c,d):

    z = torch.nan_to_num(torch.sqrt(-p), nan=0)
    v = torch.acos(q/(p*z*2.0))/3.0;
    m = torch.cos(v);
    n = torch.sin(v);
    #m = cos_acos_3( q/(p*z*2.0) );
    #n = sqrt(1.0-m*m);
        #endif
    n = n * 1.73205081;
    t = torch.clamp(torch.stack([m+m,-n-m,n-m])*z-kx,0.0,1.0).T
    res = torch.minimum( dot2(d+(c+b*t[:,0:1])*t[:,0:1]),
               dot2(d+(c+b*t[:,1:2])*t[:,1:2]) )

    return torch.nan_to_num(sel*torch.sqrt(res), nan=0)

def quad_bezier_sdf(pos, A, B, C, k=256):
    A = A.unsqueeze(0)
    B = B.unsqueeze(0)
    C = C.unsqueeze(0)
    tvals = torch.linspace(0, 1, k, device=A.device).unsqueeze(1)
    qpts = B + torch.pow(1-tvals, 2.0)*(A-B) + torch.pow(tvals, 2.0)*(C-B)
    dmat = torch.cdist(pos, qpts)
    dm, idxs = dmat.min(1)
    sel = tvals[idxs]
    selpts = qpts[idxs]
    tangents = 2*(1-sel)*(A-B) + 2*sel*(C-B)
    normals = torch.stack([-tangents[:,1], tangents[:,0]]).T
    normals = .01* (normals / torch.linalg.norm(normals, axis=1).unsqueeze(1))

    norm_dist = torch.linalg.norm(pos - (selpts+normals), axis=1)

    sign = torch.sign(norm_dist - dm)
    return sign*dm
    #quit()



def __TRASH__TO_DELETE__(aa):
    A = A.unsqueeze(0)
    B = B.unsqueeze(0)
    C = C.unsqueeze(0)

    a = B - A;
    b = A - 2.0*B + C
    c = a * 2.0
    d = A - pos

    #// cubic to be solved (kx*=3 and ky*=3)
    kk = 1.0/dot(b,b)
    kx = kk * dot(a,b)
    ky = kk * (2.0*dot(a,a)+dot(d,b))/3.0
    kz = kk * dot(d,a)

    #float res = 0.0;
    #float sgn = 0.0;

    p  = ky - kx*kx;
    q  = kx*(2.0*kx*kx - 3.0*ky) + kz;
    p3 = p*p*p;
    q2 = q*q;
    h  = q2 + 4.0*p3;

#    print([item.shape for item in [p, p3, q, h]])
#    quit()
    sel = torch.sign(torch.nn.functional.relu(h))
    case1_res = quad_bez_case1(sel,   h, p, q, kx, b,c,d)
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
