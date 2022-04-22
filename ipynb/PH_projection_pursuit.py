import torch
import bats
import time
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import directed_hausdorff
from torch_tda.nn import RipsLayer, Rips0Layer, BottleneckLayer, WassersteinLayer, BarcodePolyFeature
from torch_tda.nn import BottleneckLayerHera # torch-tda now support it
# from herabottleneck import BottleneckLayerHera
import torch.nn as nn
from torch.nn.utils.parametrizations import orthogonal
import scipy.sparse.linalg as sla
from sklearn.decomposition import PCA # for PCA analysis

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

def cayley_bottleneck_pursuit(X, lrs = np.hstack((np.repeat(1e-3,15), np.repeat(1e-4,105), np.repeat(1e-5,50))),
                              opt_dim = 1):
    Xt = torch.tensor(X, dtype=torch.double)
    layer = RipsLayer(maxdim=1, metric = 'euclidean')
    ground_truth_dgm = layer(Xt)

    pca = PCA(n_components=2).fit(X)
    P = pca.components_ # using the orthonormal matrix from PCA 
    P = P.T
    Pt = torch.tensor(P, dtype=torch.double, requires_grad=True)

    crit = BottleneckLayerHera()
    # lrs = np.hstack((np.repeat(1e-3,15), np.repeat(1e-4,105), np.repeat(1e-5,50)))
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



def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def bottleneck_proj_pursuit(X, dim=2, opt_dim=10, optimizer_iter = 10, scheduler_iter = 10, 
flags = (bats.standard_reduction_flag(),bats.clearing_flag()),
degree = +1,
PCA = False,
pca_weight = 0.5,
ortho = True,
optimizer_ = 'SGD', 
metric = 'euclidean', 
sparse = False, 
print_info = False, *args, **kwargs):
    """
    projection pursuit to minimize bottleneck distance
    
    inputs:
    --------
    X: input dataset
    dim: reduction target dimension (default 2)
    opt_dim: optimization on PH dimension 0 for H1; 1 for H1; and 10 for H1 and H0
    optimizer_iter: number of iterations for optimizer 
    scheduler_iter: number of iterations for scheduler (exponential decay for learning rate) 
    flags: BATs reduction flags used to compute PH 
        (do not use bats.extra_reduction_flag() will ruin the birth index, which is bad for opt)
    ortho: If true, we will use `from torch.nn.utils.parametrizations import orthogonal`
    optimizer_: SGD or Adam(still problematic)
    PCA: True if you want to add a PCA penalty variance with weight `pca_weight`
    sparse: True is use sparse Rips construction
    metric: supports various options of metric in BATs: L1, euclidean, etc..
    initial_weights: initial weights/projection matrix, e.g. from PCA
    
    returns:
    -----------
    projection matrix P
    optimization information opt_info
    """

    X = np.array(X, order = 'c') # necessary!!! If your data is not stored in C style
    n, p = X.shape

    linear_layer = nn.Linear(p, dim, bias=False, dtype=torch.double)
    
    # initial weights/projection matrix, e.g. from PCA
    initial_weights = kwargs.get('initial_weights', None)
    if initial_weights != None:
        linear_layer.weight = nn.Parameter(initial_weights)

    if ortho:
        model_lin = orthogonal(linear_layer)
    else: 
        model_lin = linear_layer
    
    if sparse:
        layer = RipsLayer(maxdim=1, sparse = True, eps = 0.5,
                      reduction_flags=flags)
    else:
        if opt_dim == 0:
            layer = Rips0Layer()
        else:
            layer = RipsLayer(maxdim=1, degree = degree, metric = metric, 
                              reduction_flags=flags)

    Xt = torch.tensor(X, dtype=torch.double)
    ground_truth_dgm = layer(Xt)

    # featfn = BarcodePolyFeature(0, 2,0)
    # gt_feat = featfn(ground_truth_dgm)

    # crit = BottleneckLayer()
    crit = BottleneckLayerHera()

    if optimizer_ == 'SGD':
        optimizer = torch.optim.SGD(model_lin.parameters(), lr=1e-4, momentum=0.5)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)
    if optimizer_ == 'Adam':
        optimizer = torch.optim.Adam(model_lin.parameters(), weight_decay=0.5)

    losses = []
    bd = 0

#     if print_info:
#         print('loss\t\tbd\t\tforward\t\tbottleneck\tbackward\tstep')
#     for i in tqdm(range(10)):
#         for j in tqdm(range(10)):
    lrs = []
    bd0, bd1 = torch.zeros(1), torch.zeros(1)
    for i in tqdm(range(scheduler_iter)):
        for j in range(optimizer_iter):
            ts = []
            t0_total = time.monotonic()
            optimizer.zero_grad()

            ts.append(time.monotonic())
            XVt = model_lin(Xt)
            Y_dgm = layer(XVt)
            ts.append(time.monotonic())
            if opt_dim == 10: # dim 0 and 1
                bd0 = crit(Y_dgm[0], ground_truth_dgm[0])
                bd1 = crit(Y_dgm[1], ground_truth_dgm[1])
                loss = bd0 + bd1
                bd = max(bd0.detach().numpy(), bd1.detach().numpy())
                
            if opt_dim == 1: # only dim 1
                bd1 = crit(Y_dgm[1], ground_truth_dgm[1])
                loss = bd1
                bd = bd1.detach().numpy()
            
            if opt_dim == 0: # only dim 0
                bd0 = crit(Y_dgm[0], ground_truth_dgm[0])
                loss = bd0
                bd = bd0.detach().numpy()
            
            if PCA:
                pca_layer = PCA_layer()
                pca_loss = pca_layer(XVt)
                loss -= pca_weight * pca_loss # '-': PCA layer is maximizing 
            
            ts.append(time.monotonic())

            loss.backward()
            ts.append(time.monotonic())
            optimizer.step()
            ts.append(time.monotonic())
            
            if PCA:
                losses.append([loss.detach().numpy() + pca_weight * pca_loss.detach().numpy(), 
                - pca_loss.detach().numpy()])
            else:
                losses.append(loss.detach().numpy())
            t1_total = time.monotonic()

            if print_info:
                print(f'iter {i}/{j}, loss = {loss:.4f}, time = {(t1_total - t0_total):.4f}')

            # optimizer learning rate
            lrs.append(get_lr(optimizer))

        # schedule learning rate if using SGD
        if optimizer_ == 'SGD':
            scheduler.step()
            # lrs.append(scheduler.get_last_lr())
            
        # bd0 = bd0.detach().numpy()
        # bd1 = bd1.detach().numpy()
        
    opt_info = {'bd0': bd0, 'bd1':bd1, 'losses': losses, 'lrs': lrs, 'X_dr': XVt.detach().numpy()}
    return model_lin.weight.detach().numpy(), opt_info

def subsample_bats(X, k = 100):
    '''
    sub-sampling k points, 
    input
    -----
    X: (n,p) array
        original points
    return 
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


class PCA_layer(torch.nn.Module):
    # see details in https://github.com/CompTop/PHDimensionReduction/blob/main/PCA/PCA_boots.ipynb
    def __init__(self, n_pc = 2):
        """
        Compute u^T S u as the optimization problem of PCA.
        
        Arguments:
            p: original dataset feature dimension
            n_pc: number of principal components or dimension of projected space,
                  defaulted to be 2
        """
        super().__init__()
        
    def forward(self, XV):
        """
        XV: X @ V, where V is the orthornormal column matrix
        """
        n = XV.shape[0]
        return (1/n * torch.pow(torch.linalg.norm(XV, 'fro'), 2) )



#TODO LBFGS will cause -1 index in BottleneckDistanceHera.forward()
def bot_proj_pursuit_LBFGS(X, dim=2, opt_dim=10, metric = 'euclidean', 
                           sparse = False, print_info=False):
    """
    projection pursuit to minimize bottleneck distance using LBFGS as optimzer
    
    inputs:
    --------
    X: input dataset
    dim: reduction target dimension (default 2)
    opt_dim: optimization on PH dimension 0 for H1; 1 for H1; and 10 for H1 and H0
    
    returns:
    -----------
    projection matrix P
    bottleneck distance
    """
    X = np.array(X, order = 'c')
    n, p = X.shape

    model_lin = orthogonal(nn.Linear(p, dim, bias=False, dtype=torch.double))
    
    if sparse:
        layer = RipsLayer(maxdim=1, sparse = True, eps = 0.5,
                      reduction_flags=(bats.standard_reduction_flag(), 
                                                 bats.clearing_flag(),))
    else:
        if opt_dim == 0:
            layer = Rips0Layer()
        else:
            layer = RipsLayer(maxdim=1, metric = metric,
                              reduction_flags=(bats.standard_reduction_flag(), 
                                               bats.clearing_flag(),))

    Xt = torch.tensor(X, dtype=torch.double)
    ground_truth_dgm = layer(Xt)
    crit = BottleneckLayerHera()
    optimizer = torch.optim.LBFGS(model_lin.parameters(), lr=1e-2, max_iter=20, 
                                  max_eval=None, tolerance_grad=1e-07,
                                  tolerance_change=1e-09, 
                                  history_size=8, line_search_fn='strong_wolfe')
    
    losses = []
    bd = 0

    def closure():
        optimizer.zero_grad()
        Y_dgm = layer(model_lin(Xt))

        if opt_dim == 10: # dim 0 and 1
            bd0 = crit(Y_dgm[0], ground_truth_dgm[0])
            bd1 = crit(Y_dgm[1], ground_truth_dgm[1])
            loss = bd0 + bd1
            bd = max(bd0.detach().numpy(), bd1.detach().numpy())

        if opt_dim == 1: # only dim 1
            bd1 = crit(Y_dgm[1], ground_truth_dgm[1])
            loss = bd1
            bd = bd1.detach().numpy()

        if opt_dim == 0: # only dim 0
            bd0 = crit(Y_dgm[0], ground_truth_dgm[0])
            loss = bd0
            bd = bd0.detach().numpy()
            
        if print_info:
                print(f'iter, loss = {loss:.4f}')
            
        loss.backward()
        losses.append(loss.detach().numpy())
        return loss
    
    lrs = []
    for i in range(5):
        for j in range(10):
            optimizer.step(closure)
            lrs.append(get_lr(optimizer))
    
#         scheduler.step()
#         lrs.append(scheduler.get_last_lr())
            
    return model_lin.weight.detach().numpy(), bd, losses