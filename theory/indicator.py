# -*- coding: utf-8 -*-
"""

@author: dgarreau

All functions related to indicator functions are collected in this file.

"""

import numpy as np

from scipy.stats import norm

from utils.aux_functions import get_middle_point

from theory.general import get_bxi
from theory.general import compute_beta_multiplicative
from theory.general import get_bin_indices

def compute_beta_indicator(xi,rect,nu,my_stats):
    """
    Compute beta for an indicator function with rectangular support. 
    
    INPUT:
        xi: example to explain (size (d,))
        rect: support of the indicator function (size (d,2))
        nu: bandwidth parameter 
        my_stats: summary statistics (see get_training_data_stats)
        
    OUTPUT:
        beta: theoretical value of the explanations
    
    """
    ews = compute_ew_indicator(xi,rect,nu,my_stats)
    beta = compute_beta_multiplicative(xi,nu,ews,my_stats)
    return beta

def compute_beta_indicator_large_nu(xi,rect,my_stats):
    """
    This function computes an approximation of beta for a given hyper-rectangle 
    A under two assumptions:
        (i) A is strictly included in a bin;
        (ii) nu is large.
    
    INPUT:
        xi: example to explain (size (d,))
        rect: support of the indicator function (size (d,2))
        my_stats: summary statistics (see get_training_data_stats)
        
    OUTPUT:
        beta: theoretical value of the explanations (size (dim+1,))
    """
    dim = len(my_stats["means"])
    p = len(my_stats["mins"][0])
    beta = np.zeros((dim+1,))
    bin_indices = get_bin_indices(rect,my_stats)
    
    # compute the relative importance of A
    imp = compute_relative_importance(rect,my_stats)
    
    for j in range(dim):
        
        b = bin_indices[j]
        qinf = my_stats["mins"][j][b]
        qsup = my_stats["maxs"][j][b]
        
        # check if \xi_j is in the bin containing A
        if qinf <= xi[j] and qsup >= xi[j]:
            beta[j+1] = imp / p**(dim-1)
        else:
            beta[j+1] = -imp / ((p-1)*p**(dim-1))
    
    # compute the intercept
    beta[0] = imp/p**dim - (1/p) * np.sum(beta)
    return beta

def compute_ew_indicator(xi,rect,nu,my_stats):
    """"
    Compute e^{f_j}_{j,b} for an indicator function according to Eq. (12) in 
    the paper. 

    INPUT:
        xi: example to explain (size (dim,))
        rect: support of the indicator function (size (dim,2))
        nu: bandwidth parameter (positive)
        my_stats: summary statistics (see get_training_data_stats)
        
    OUTPUT:
        ews: array containing the e^{f_j}_{j,b} coefficients (size (dim,p))
        
    """
    dim = xi.shape[0]
    p = len(my_stats["mins"][0])
    bxi = get_bxi(xi,my_stats)
    ews = np.zeros((dim,p))

    # for each dimension
    for j in range(dim):
        
        # get the limits of the hyper-rectangle A
        left = rect[j,0]
        right = rect[j,1]
        
        # for each bin along this dimension
        for b in range(p):
            
            # get the bin limits
            qinf = my_stats["mins"][j][b]
            qsup = my_stats["maxs"][j][b]

            # get the mean and std on this bin
            mu = my_stats["means"][j][b]
            sigma = my_stats["stds"][j][b]
                
            rup = (np.min([right,qsup]) - mu) / sigma
            lup = (np.max([left,qinf]) - mu) / sigma
            
            # intersection is empty, return zero
            if rup < lup:
                aux = 0
            # intersection is not empty, formula (12) in the paper
            else:
                ldown = (qinf - mu) / sigma
                rdown = (qsup - mu) / sigma
                aux = (norm.cdf(rup)-norm.cdf(lup)) / (norm.cdf(rdown)-norm.cdf(ldown))
                
            if b == bxi[j]:
                ews[j,b] = aux
            else:
                ews[j,b] = aux * np.exp(-1/(2*nu**2)) 
    
    return ews

def compute_relative_importance(rect,my_stats):
    """
    This function computes the relative importance of a rectangle included in 
    a d-dimensional bin (see Definition 1 in the paper).
    
    INPUT:
        rect: a rectangle (size (dim,2))
        my_stats: summary statistics (see get_training_data_stats)
        
    OUTPUT:
        relative importance
        
    """
    dim = len(my_stats["means"])
    lbdas = np.zeros((dim,))
    bin_indices = get_bin_indices(rect,my_stats)
    for j in range(dim):
            # get the bin index containing A along this dimension
            b = bin_indices[j]
        
            # get the bin limits
            qinf = my_stats["mins"][j][b]
            qsup = my_stats["maxs"][j][b]

            # get the mean and std on this bin
            mu = my_stats["means"][j][b]
            sigma = my_stats["stds"][j][b]
                
            rup = (rect[j,1] - mu) / sigma
            lup = (rect[j,0] - mu) / sigma
            rdown = (qsup - mu) / sigma
            ldown = (qinf - mu) / sigma
         
            lbdas[j] = (norm.cdf(rup) - norm.cdf(lup)) / (norm.cdf(rdown) - norm.cdf(ldown))

    return np.prod(lbdas)
    
def compute_bin_dist(xi,rect,my_stats):
    """
    Computes the bin distance between xi and a given rectangle included in a 
    d-dimensional bin (see Definition 2 in the paper).
    
    INPUT:
        xi: example to explain
        rect: a rectangle (size (dim,2))
        my_stats: summary statistics (see get_training_data_stats)

    OUTPUT:
         bin distance between xi and rect
         
    """
    dim = len(my_stats["means"])
    mid_point = get_middle_point(rect)
    
    bin_indices = get_bxi(mid_point,my_stats)
    count = 0
    for j in range(dim):
            # get the bin index containing A along this dimension
            b = bin_indices[j]
        
            # get the bin limits
            qinf = my_stats["mins"][j][b]
            qsup = my_stats["maxs"][j][b]
            
            # count 1 if not in the same bin
            if xi[j] < qinf or xi[j] > qsup:
                count += 1
                
    return count
    
    
