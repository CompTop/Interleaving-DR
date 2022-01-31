import scipy.sparse.linalg as sla
import numpy as np

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