import numpy as np
from numpy.linalg import norm

from scipy.stats import norm as gaussian

def ones(n):
    return np.ones((n,1), dtype = float)

def diag(A):
    # Zeros matrix except for diagonal
    return np.diag(np.diag(A))

def rbf(X,Y,sigma=1):
    m = X.shape[0]
    assert m == Y.shape[0]

    K = np.empty((m,m))

    # TODO: Make numpy handle this so that C calculates K
    for i in range(m):
        for j in range(m):
            K[i,j] = np.exp(-sigma**2 * norm(X[i]-X[j])**2)
    return K

def mn(m,n):
    return np.math.factorial(m) / np.math.factorial(m-n)

def var_HSIC_b(X,Y):
    # Variance of HSIC_b under H0
    m = X.shape[0]
    assert m == Y.shape[0]

    # TODO: pass along correct sigma
    K = rbf(X,X)
    L = rbf(Y,Y)

    H = np.eye(m) - np.outer(ones(m),ones(m)) / m

    HKH = H @ K @ H
    HLH = H @ L @ H

    B = (HKH * HLH)**2

    return ( 2 * (m-4) * (m-5) / mn(m,4) )   *   \
                   ( ones(m).T @ (B - diag(B)) @ ones(m) ) **2

def expected_HSIC_b(X,Y):
    # Expected of HSIC_b under H0
    m = X.shape[0]
    assert m == Y.shape[0]

    # TODO: If it doesn't work, try by subtracting K(i,i)'s
    sp_mu_x_hat = 1/mn(m,2) * ones(m).T @ K @ ones(m)
    return (1 + sp_mu_x_hat * sp_mu_y_hat - sp_mu_x_hat - sp_mu_y_hat) / m

def HSIC_b(X,Y):
    m = X.shape[0]
    assert m == Y.shape[0]

    H = np.eye(m) - (ones(m) @ ones(m).T) / m
    # TODO: pass along correct sigma
    K = rbf(X,X)
    L = rbf(Y,Y)

    return np.trace(K @ H @ L @ H) / m**2

def indep_test(X, Y, alpha=0.05):
    # Construct confidence interval from fact that:
    #                             P(C*\HSIC_hat-HSIC\>z_{1-alpha/2}) --> 1-alpha
    # And conclude that normality assumption can be rejected (hence dependence)
    # if HSIC_b_hat does not actually lie in confidence interval
    E_HSIC_b = expected_HSIC_b(X,Y)
    var_HSIC = var_HSIC_b(X,Y)
    realization = HSIC_b(X,Y)
    lower = E_HSIC_b - var_HSIC * gaussian.ppf(1-alpha/2)
    upper = E_HSIC_b + var_HSIC * gaussian.ppf(1-alpha/2)
    if  lower < realization < upper :
        return
