# -*- coding: utf-8 -*-
"""

@author: dgarreau

All functions related to kernels are colloected here.

"""

import numpy as np

from scipy.stats import norm

from theory.general import get_bxi
from theory.general import compute_beta_multiplicative

def compute_ew_gk(xi,nu,zeta,gamma,my_stats):
    """"
    Compute e^{f_j}_{j,b} for the Gaussian kernel with default weights. See 
    Eq. (15) in the paper.
    
    INPUT:
        xi: example to explain (size (dim,))
        nu: bandwidth parameter
        zeta: location parameter of the Gaussian kernel (size (dim,))
        gamma: bandwidth parameter of the Gaussian kernel 
        my_stats: summary statistics (see get_training_data_stats)
    
    OUTPUT:
        ews: matrix containing the coefficients, size (dim,p)
            
    """
    dim = xi.shape[0]
    p = len(my_stats["mins"][0])
    
    # get the bin indices of xi
    bxi = get_bxi(xi,my_stats)
    
    ews = np.zeros((dim,p))
    for j in range(dim):
        for b in range(p):
            
            # for each k,p, straightforward computation from Eq. (15)
            mu = my_stats["means"][j][b]
            sigma = my_stats["stds"][j][b]
            qinf = my_stats["mins"][j][b]
            qsup = my_stats["maxs"][j][b]
            mtilde = (gamma**2 * mu + sigma**2*zeta[j]) / (gamma**2 + sigma**2)
            stilde = sigma*gamma / np.sqrt(gamma**2 + sigma**2)
            ll = (qinf - mu) / sigma
            rr = (qsup - mu) / sigma
            lt = (qinf - mtilde) / stilde
            rt = (qsup - mtilde) / stilde
            aux = (stilde / sigma) * (norm.cdf(rt)-norm.cdf(lt)) *np.exp(-(mu-zeta[j])**2/(2*(gamma**2+sigma**2))) / (norm.cdf(rr)-norm.cdf(ll))
            
            # weighting if in the special bin
            if b == bxi[j]:
                ews[j,b] = aux
            else:
                ews[j,b] = aux * np.exp(-1/(2*nu**2))
                
    return ews

def compute_beta_gk(xi,nu,zeta,gamma,my_stats):
    """
    Compute beta for the Gaussian kernel with default weights, see 
    Proposition 5 in the paper.
    
    INPUT:
        xi: example to explain (size (dim,))
        nu: bandwidth parameter
        zeta: location parameter of the Gaussian kernel (size (dim,))
        gamma: bandwidth parameter of the Gaussian kernel 
        my_stats: summary statistics (see get_training_data_stats)
        
    OUTPUT:
        theoretical interpretable coefficients, size (dim+1,)
        
    """
    ews = compute_ew_gk(xi,nu,zeta,gamma,my_stats)
    beta = compute_beta_multiplicative(xi,nu,ews,my_stats)
    return beta
