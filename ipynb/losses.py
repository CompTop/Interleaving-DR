import torch
from torch_tda.nn import RipsLayer, Rips0Layer, BottleneckLayer, WassersteinLayer, BarcodePolyFeature, BottleneckLayerHera

def diagram_loss(X, maxdim=1, bottleneck_wts=None, wasserstein_wts=None):
    """
    produce loss on persistence diagram of X
    
    Inputs:
        X: reference point cloud
        maxdim: maximum homology dimension (Default 1)
        bottleneck_wts: weights for bottleneck distance in each homology dimension.  Default will weight each equally
        wasserstein_wts: weights for wasserstein distance in each homology dimension. Default is zero.
        
    Outputs:
        Loss function that can be called on input point cloud represented as torch tensor
    
    """
    layer = RipsLayer(maxdim=maxdim, metric='euclidean', degree=+1,
                     reduction_flags=(bats.standard_reduction_flag(), bats.clearing_flag())
                     )
    
    Xt = torch.tensor(X, dtype=torch.double)
    X_dgm = layer(Xt)
    dB = BottleneckLayerHera()
    dW = WassersteinLayer()
    if bottleneck_wts is None:
        bottleneck_wts = [1 for i in range(maxdim+1)]
    if wasserstein_wts is None:
        wasserstein_wts = [0 for i in range(maxdim+1)]
    
    def loss(Y):
        Y_dgm = layer(Y)
        l = 0
        for i, w in enumerate(bottleneck_wts):
            if w:
                l = l + w * dB(X_dgm[i], Y_dgm[i])
        
        for i, w in enumerate(wasserstein_wts):
            if w:
                l = l + w * dW(X_dgm[i], Y_dgm[i])
        
        return l
        
    
    return loss

def pca_loss(X):
    n = X.shape[0]
    return (1/n * torch.pow(torch.linalg.norm(X, 'fro'), 2))