# -*- coding: utf-8 -*-
"""

@author: dgarreau

In this file are collected the functions used in the various theoretical 
computations.

"""

import numpy as np

from scipy.stats import norm

def compute_little_c(p,nu):
    """
    Computes the ubiquitous c constant.
    
    INPUT:
        p: number of bins 
        nu: bandwidth parameter
        
    OUTPUT:
        c defined by Eq. (4) in the paper
    """
    return 1/p + (1-1/p)*np.exp(-1/(2*nu**2))

def compute_mutilde(my_stats):
    """
    This function computes the mean of the truncated Gaussians on all the 
    d-dimensional bins.
    
    INPUT:
        my_stats: summary statistics (see get_training_data_stats)
        
    OUTPUT:
        mutilde (size (dim,p))
        
    """

    dim = len(my_stats["means"])
    p = len(my_stats["mins"][0])
    
    mutilde = np.zeros((dim,p))
    for j in range(dim):
        for b in range(p):
            mu    = my_stats["means"][j][b]
            sigma = my_stats["stds"][j][b]
            left  = (my_stats["mins"][j][b] - mu) / sigma
            right = (my_stats["maxs"][j][b] - mu) / sigma
            
            mutilde[j,b] = mu + (norm.pdf(left) - norm.pdf(right)) / (norm.cdf(right) - norm.cdf(left))
    
    return mutilde

def compute_sigmatilde(my_stats):
    """
    This function computes the standard deviations of the truncated Gaussian 
    on all the d-dimensional bins.
    
    INPUT:
        my_stats: summary statistics (see get_training_data_stats)
        
    OUTPUT:
        sigmatilde (size (dim,p))
        
    """

    dim = len(my_stats["means"])
    p = len(my_stats["mins"][0])
    
    sigmatilde = np.zeros((dim,p))
    for j in range(dim):
        for b in range(p):
            mu    = my_stats["means"][j][b]
            sigma = my_stats["stds"][j][b]
            left  = (my_stats["mins"][j][b] - mu) / sigma
            right = (my_stats["maxs"][j][b] - mu) / sigma
            
            Z = norm.cdf(right) - norm.cdf(left)
            
            sigmatilde[j,b] = sigma * np.sqrt(1 + (left * norm.pdf(left) - right * norm.pdf(right)) / Z - ((norm.pdf(left) - norm.pdf(right))/Z)**2)
    
    return sigmatilde

def get_bxi(xi,my_stats):
    """
    This function gets the bin indices for xi given summary statistics.
    
    INPUT
        xi: example to explain (size (dim,))
        my_stats: summary statistics (see get_training_data_stats)
        
    OUTPUT:
        bxi: bin indices (size (dim,))
    
    """
    p = len(my_stats["mins"][0])
    dim = xi.shape[0]
    
    bxi = np.zeros((dim,),dtype=int)
    for j in range(dim):
        for b in range(p):
            # b = bxi_j if xi_j belongs to the bin
            if my_stats["mins"][j][b] <= xi[j] and xi[j] <= my_stats["maxs"][j][b]:
                bxi[j] = b
                break
            
    return bxi


def get_bin_indices(rect,my_stats):
    """
    This function gets the bins indices of a rectangle A.
    
    INPUT:
        rect: array with left and right limit of the rectangle for each 
        dimension (size (d,2))
        my_stats: summary statistics (see get_training_data_stats)
        
    OUTPUT:
        bin_indices: array containing the bin indices along each dimension 
        (size (d,))
        
    """
    dim = len(my_stats["means"])
    p = len(my_stats["mins"][0])
    
    bin_indices = np.zeros((dim,),dtype=int)
    for j in range(dim):
        
        # get the limits of A along j
        left = rect[j,0]
        right = rect[j,1]
        
        found = False
        for b in range(p):
            
            # get the bin limits 
            qinf = my_stats["mins"][j][b]
            qsup = my_stats["maxs"][j][b]
            
            # test
            if qinf <= left and qsup >= right:
                bin_indices[j] = b
                found = True
                break
        if not found:
            raise ValueError("A is not included in a bin!")
            
    return bin_indices


def compute_beta_additive(xi,nu,ews,my_stats):
    """
    Compute beta for an additive model if provided with the e_{jb}^{f_j}
    coefficients according to Proposition 3 in the paper.
    
    INPUT:
        xi: example to explain (size (dim,))
        nu: bandwidth parameter
        ews: e_{j,b}^{f_j} coefficients corresponding to f
        my_stats: summary statistics, see get_training_data_stats
        
    OUTPUT:
        beta: theoretical explanation (size (dim+1,))
    
    """
    dim = xi.shape[0]
    p = len(my_stats["mins"][0])
    
    # get the bin indices
    bxi = get_bxi(xi,my_stats)
    
    # compute the c_j^{f_j}
    cf_store = np.mean(ews,1)
    
    # normalization constants
    little_c = compute_little_c(p,nu)

    # computation of beta
    beta = np.zeros((dim+1,))
    aux = 0
    for j in range(dim):
        aux += p*cf_store[j] - ews[j,bxi[j]]
        beta[j] = (p*little_c / (p*little_c-1)) * (ews[j,bxi[j]] - cf_store[j] / little_c)
        
    beta[0] = (1/(p*little_c - 1)) * aux
    
    return beta

def compute_beta_multiplicative(xi,nu,ews,my_stats):
    """
    Compute beta for a multiplicative model if provided with the ew 
    coefficients according to Proposition 5 in the paper.
    
    INPUT:
        xi: example to explain (size (dim,))
        nu: bandwidth parameter
        ews: e_{j,b}^{f_j} coefficients corresponding to f
        my_stats: summary statistics, see get_training_data_stats
        
    OUTPUT:
        beta: theoretical explanation (size (dim+1,))
    
    """
    dim = xi.shape[0]
    p = len(my_stats["mins"][0])
    
    # get the bin indices
    bxi = get_bxi(xi,my_stats)
    
    # compute the c_j^{f_j}
    cf_store = np.mean(ews,1)
    
    # normalization constants
    little_c = compute_little_c(p,nu)
    big_c = little_c**dim
    prod_cst = np.prod(cf_store) / big_c

    # computation of beta
    beta = np.zeros((dim+1,))
    for j in range(dim):
        
        aux = ews[j,bxi[j]] * little_c / cf_store[j]- 1
        beta[0] -= (1 / (p*little_c - 1)) * aux
        beta[j+1] = prod_cst * (p*little_c / (p*little_c - 1)) * aux
    beta[0] = prod_cst * (1+beta[0])
    
    return beta
