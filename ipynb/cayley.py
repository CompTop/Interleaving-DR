import scipy.sparse.linalg as sla
import bats
import time
from tqdm import tqdm
import torch
from torch_tda.nn import RipsLayer, Rips0Layer, BottleneckLayer, WassersteinLayer, BarcodePolyFeature
from herabottleneck import BottleneckLayerHera
import glob
import numpy as np
import matplotlib.pyplot as plt
import bats
from sklearn.metrics.pairwise import pairwise_distances


def cayley_update(V, dV, tau):
    """
    update V in dV direction with step size tau
    
    use lsmr with linear operator
    
    returns:
        X: new orthonormal matrix after update
    """
    n, p = V.shape
    dVV = dV.T @ V
    VV = V.T @ V
    V2 = V - tau * (dV @ VV - V @ dVV)
    dVVTop = sla.LinearOperator(
        shape=(n,n),
        matvec = lambda A : dV @ (V.T @ A),
        rmatvec = lambda A : V @ (dV.T @ A),
        matmat = lambda A : dV @ (V.T @ A),
    )
    C2op = sla.LinearOperator(
        shape=(n,n),
        matvec = lambda A : A + tau * (dVVTop.matvec(A) - dVVTop.rmatvec(A)),
        rmatvec = lambda A : A + tau * (dVVTop.rmatvec(A) - dVVTop.matvec(A))
        #matmat = lambda A : A + tau * (dVVTop.matmat(A) - dVVTop.T.matmat(A)),
        #rmatmat = lambda A : A + tau * (dVVTop.rmatmat(A) - dVVTop.matmat(A))
    )
    X = np.empty((n,p))
    for j in range(p):
        xj, istop, itn, normr = sla.lsmr(C2op, V2[:,j])[:4]
        X[:,j] = xj
    return X


def cayley_bottleneck_pursuit(X, 
                              lrs = np.hstack((np.repeat(1e-3,15), np.repeat(1e-4,50), np.repeat(1e-5,35))),
                              opt_dim = 1):
    '''
    lrs: learning rates
    
    '''
    
    Xt = torch.tensor(X, dtype=torch.double)
    layer = RipsLayer(maxdim=1, metric = 'euclidean')
    ground_truth_dgm = layer(Xt)

    pca = PCA(n_components=2).fit(X)
    P = pca.components_ # using the orthonormal matrix from PCA 
    P = P.T
    Pt = torch.tensor(P, dtype=torch.double, requires_grad=True)

    crit = BottleneckLayerHera()
    
    losses = []
    bd0, bd1 = torch.zeros(1), torch.zeros(1)
    for lr in tqdm(lrs):
        Yt = torch.mm(Xt, Pt)
        Y_dgm = layer(Yt)
        
        if opt_dim == 1:
            bd1 = crit(Y_dgm[1], ground_truth_dgm[1])
            loss = bd1
        elif opt_dim == 10:
            bd0 = crit(Y_dgm[0], ground_truth_dgm[0])
            bd1 = crit(Y_dgm[1], ground_truth_dgm[1])
            loss = bd0 + bd1
        else:
            bd0 = crit(Y_dgm[0], ground_truth_dgm[0])
            loss = bd0
            
        losses.append(loss.detach().numpy())
    #     print(loss.detach().numpy())

        try:
            Pt.grad.zero_()
        except:
            pass

        loss.backward()

        # detach from torch to do update
        dP = Pt.grad.detach().numpy()
        P = cayley_update(P, dP, lr)
        # put back in tensor
        Pt.data = torch.tensor(P)
    
    opt_info = {'bd0': bd0, 'bd1':bd1, 'losses': losses}
    return Pt, opt_info

def subsample_bats(X, k = 100):
    '''
    sub-sampling k points
    
    
    Input:
    -----
    X: (n,p) array
        original points
        
    Output: 
    -----
    Xk: (k,p) array
        subsampled points 
        
    new_inds: (k,) array
        indices of subsampled pts, can be used for color labelling in plot
    dHX: float
        directed Hausdorff distance between Xk and X
    '''
    X = np.array(X, order = 'c')
    Xbats = bats.DataSet(bats.Matrix(X))
    inds, dists = bats.greedy_landmarks_hausdorff(Xbats, bats.Euclidean(), 0)
    new_inds = inds[:k+1]
    Xk = X[new_inds]
    dHX = dists[k]
    return Xk, new_inds, dHX


def cayley_opt(X, crit, P=None, lrs=None):
    """
    Optimize projection
    
    Inputs:
        X: data
        crit: criterion to minimize.
        P: Initial projection.  If None, will initialize with PCA
        lrs: sequence of learning rates
        crit: criterion to minimize
        
    Returns:
        P: projection
        losses: sequence of losses
        
        
    example:
    ```
    dloss = diagram_loss(Y, bottleneck_wts=[0,1], wasserstein_wts=[0,0.01])
    crit = lambda Y : dloss(Y) + 100*pca_loss(Y)
    P, losses, Ps = cayley_opt(Y, crit, P=P, lrs=np.hstack((np.repeat(3e-3,23),)))
    ```
    
    """
    if P is None:
        pca = PCA(n_components=2).fit(X)
        P = pca.components_
        P = P.T
        
    if lrs is None:
        lrs = np.hstack((np.repeat(1e-3,15), np.repeat(1e-4,65), np.repeat(1e-5,10)))
        
    Xt = torch.tensor(X, dtype=torch.double)
    Pt = torch.tensor(P, dtype=torch.double, requires_grad=True)
    
    
    losses = []
    Ps = [np.array(P, copy=True)]

    for lr in tqdm(lrs):
        Yt = torch.mm(Xt, Pt)

        loss = crit(Yt)
        losses.append(loss.detach().numpy())

        try:
            Pt.grad.zero_()
        except:
            pass

        loss.backward()

        # detach from torch to do update
        dP = Pt.grad.detach().numpy()
        P = cayley_update(P, dP, lr)
        Ps.append(np.array(P, copy=True))
        # put back in tensor
        Pt.data = torch.tensor(P)
        
    
    # append loss
    Yt = torch.mm(Xt, Pt)
    loss = crit(Yt)
    losses.append(loss.detach().numpy())
    
        
    return Pt.detach().numpy(), losses, Ps
