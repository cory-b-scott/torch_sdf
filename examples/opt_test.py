import numpy as np
import torch

from src.sdfs import *
from src.binary_ops import *
from src.unary_ops import *
from src.functional_sdfs import *

import matplotlib.pyplot as plt
import matplotlib as mpl
from skimage.measure import find_contours

from skimage.transform import resize as skresize
from skimage.io import imread, imsave
from scipy.ndimage import distance_transform_edt

from test import get_image_coords

class MultiCircleSDF(torch.nn.Module):

    def __init__(self, n_circs, device='cpu'):
        super(MultiCircleSDF, self).__init__()
        tparams = [torch.nn.Parameter(.5-torch.rand((2,)).to(device)) for i in range(n_circs)]
        rads = [torch.nn.Parameter(.1*torch.rand((1,)).to(device)) for i in range(n_circs)]
        self.circs = torch.nn.ModuleList([
            TranslatedSDF(t,SphereSDF(r)) for t,r in zip(tparams, rads)
        ])
        self.k = torch.tensor(1.0/24.0).to(device)
        self.device = device


    def forward(self, query):
        dists = torch.stack([sdf(query) for sdf in self.circs])
        return binops.nary_smooth_union_exp(dists, k=self.k)



class MultiRectSDF(torch.nn.Module):

    def __init__(self, n_circs, device='cpu'):
        super(MultiRectSDF, self).__init__()
        tparams = [torch.nn.Parameter(.5-torch.rand((2,)).to(device)) for i in range(n_circs)]
        rads = [torch.nn.Parameter(.1*torch.rand((2,)).to(device)) for i in range(n_circs)]
        angles = [torch.nn.Parameter(.1*torch.rand((1,)).float().to(device)) for i in range(n_circs)]
        self.circs = torch.nn.ModuleList([
            RotatedSDF(
                an,
                torch.tensor([[1.0],[0.0]]).to(device),
                torch.tensor([[0.0],[1.0]]).to(device),
                TranslatedSDF(t,RectSDF(r)))

            for t,r,an in zip(tparams, rads, angles)
        ])
        self.device = device


    def forward(self, query):
        dists = torch.stack([sdf(query) for sdf in self.circs])
        return binops.nary_sharp_union(dists)

def train_step(model, opt, criterion, loader):
    total_loss = 0.0
    for bX, bY in loader:

        opt.zero_grad()

        preds = model(bX.to(dev))
        loss = criterion(preds, bY.to(dev))

        loss.backward()
        total_loss += loss*bX.shape[0]
        opt.step()

    return total_loss.detach().cpu().numpy() / len(loader.dataset)

def sdf_plot(sdf_arr, ax, with_contours=False):
    print(sdf_arr.min(), sdf_arr.max())
    divnorm=mpl.colors.TwoSlopeNorm(vmin=sdf_arr.min(), vcenter=0., vmax=sdf_arr.max())

    ax.imshow(sdf_arr, cmap=mpl.colormaps['RdBu'], norm=divnorm)
    if with_contours:
        contours = find_contours(sdf_arr, 0.0)
        for contour in contours:
            ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
    ax.axis('image')
    ax.set_aspect(1.0)
    ax.axis('off')

def visualize(model, loader, K, step=0):
    all_distances = []
    all_targets = []
    with torch.no_grad():
        for i,(bX,bY) in enumerate(loader):
            preds = model(bX.to(model.device))
            #if i == 0:
            #    print(preds[:10], bY[:10])
            all_distances.append(preds.detach().cpu().numpy())
            all_targets.append(bY.numpy())
    dists = np.concatenate(all_distances)
    targs = np.concatenate(all_targets)

    dn = dists.reshape((K,K))#[::-1, ::-1]
    tn = targs.reshape((K,K))

    fig,ax = plt.subplots(1,2)

    sdf_plot(dn, ax[0])
    sdf_plot(tn, ax[1])

    fig.tight_layout()
    fig.savefig("outputs/sdf_learn_%03d.png" % step )

if __name__ == '__main__':

    K = 256

    dev = 'cuda'

    target = imread("squares.png")
    target = 1 - np.sign(skresize(target, (K,K)))
    target = distance_transform_edt(target) - distance_transform_edt(1.0 - target)
    target /= K

    imcoords, imcenter, imscale = get_image_coords(K)

    model = MultiRectSDF(3, device=dev)
    opt = torch.optim.Adam(model.parameters(), 0.001)

    criterion = torch.nn.MSELoss()

    X = torch.tensor(imcoords)
    Y = torch.tensor(target.reshape(-1,))

    print(X.shape, Y.shape)

    data = torch.utils.data.TensorDataset(X.float(),Y.float())
    shuf_loader = torch.utils.data.DataLoader(data, batch_size=10024, shuffle=True, drop_last=False)
    order_loader = torch.utils.data.DataLoader(data, batch_size=10024, shuffle=False, drop_last=False)

    for ep in range(200):

        ep_loss = train_step(model, opt, criterion, shuf_loader)
        print(ep_loss)
        if ep % 1 == 0:
            visualize(model, order_loader, K, step=ep)
