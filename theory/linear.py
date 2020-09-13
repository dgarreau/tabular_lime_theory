# -*- coding: utf-8 -*-
"""

@author: dgarreau

In this file we collect all functions related to theoretical explanations for 
linear models.

"""

import numpy as np

from theory.general import compute_mutilde
from theory.general import get_bxi

def compute_beta_linear(xi,f,nu,my_stats):
    """
    This function computes \beta^f when f is linear according to Corollary 4 
    in the paper.
    
    INPUT:
        xi: example to explain (size (dim,))
        f: coefficients of the function to explain, with intercept in first 
        position (size (dim+1,))
        nu: bandwidth parameter
        my_stats: summary statistics (see get_training_data_stats)
        
    OUTPUT:
        beta: theoretical explanation (size (dim+1,))
    
    """
    p = len(my_stats["mins"][0])
    dim = xi.shape[0]
    
    # means of the truncated Gaussians
    mutilde = compute_mutilde(my_stats)
    
    # bin indices of \xi
    bxi = get_bxi(xi,my_stats)
    
    # computing \muttilde
    exps = np.exp(-1/nu**2)
    muttilde = np.zeros((dim,))
    for j in range(dim):
        muttilde[j] = (exps * np.sum(mutilde[j,:]) + (1-exps) * mutilde[j,bxi[j]]) / (1 + (p-1)*exps)
    
    # computing beta
    aux = f[0] + np.dot(f[1:],muttilde)
    for j in range(dim):
        aux -= f[j+1] * (mutilde[j,bxi[j]] - muttilde[j]) / ((p-1)*exps)
    beta = np.zeros((dim+1,))
    beta[0] = aux
    for j in range(dim):
        beta[j+1] = (f[j+1] / (p-1)) * np.sum(mutilde[j,bxi[j]] - mutilde[j,:])
    
    return beta

