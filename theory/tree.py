# -*- coding: utf-8 -*-
"""

@author: dgarreau

In this file are collected all tree-related functions.

"""

import numpy as np

from utils.aux_functions import get_empirical_boundaries
from utils.aux_functions import get_partition    
from utils.aux_functions import get_middle_point
from utils.aux_functions import split_partition

from theory.indicator import compute_beta_indicator
from theory.indicator import compute_beta_indicator_large_nu

def compute_beta_tree(xi,nu,tree_regressor,my_stats,verbose=False):
    """
    Computes the theoretical explanation for a tree with rectangular cells by 
    browsing the partition and computing the indicator explanation for each 
    rectangle. This can take a while when dimension and depth are not small. 
    
    INPUT:
        xi: example to explain (size (dim,))
        nu: bandwidth parameter
        tree_regressor: sklearn.tree.tree.DecisionTreeRegressor object
        my_stats: summary statistics (see get_training_data_stats)
        verbose: set to True to follow the progress
        
    OUTPUT:
        beta of size (dim+1,)
    
    """
    dim = len(my_stats["means"])
    
    # get the empirical boundaries
    emp_boundaries = get_empirical_boundaries(my_stats)
    
    # get the partition of the input space corresponding to the tree regressor
    partition = get_partition(tree_regressor,emp_boundaries)
    
    idx = 0
    size_partition = len(partition)
    alpha_store = np.zeros((size_partition,))
    beta_aux_store = np.zeros((size_partition,dim+1))
    
    # let us go through the partition
    for rect in partition:
        
        if verbose and np.mod(idx,100) == 0:
            print("Treating rectangle number {} / {}...".format(idx+1,size_partition))
            
        # get the value of f(A)
        mid = get_middle_point(rect)
        alpha_store[idx] = tree_regressor.predict(mid.reshape(1, -1))
        
        # get the explanation for A
        beta_aux_store[idx] = compute_beta_indicator(xi,rect,nu,my_stats)
        idx += 1
    
    # final result is the weighted sum
    beta_theo = alpha_store.dot(beta_aux_store)
    
    return beta_theo
 
def compute_beta_tree_large_nu(xi,tree_regressor,my_stats,verbose=False): 
    """
    Computes the theoretical explanation for a tree with rectangular cells 
    when the bandwidth parameter is large.
    
    INPUT:
        xi: example to explain (size (dim,))
        tree_regressor: sklearn.tree.tree.DecisionTreeRegressor object
        my_stats: summary statistics (see get_training_data_stats)
        verbose: set to True to follow the progress
        
    OUTPUT:
        beta of size (dim+1,)
    
    """
    dim = len(my_stats["means"])
    
    # get the empirical boundaries
    emp_boundaries = get_empirical_boundaries(my_stats)
    
    # get the partition of the input space corresponding to the tree regressor
    partition = get_partition(tree_regressor,emp_boundaries)
    
    # first intersect the partition with the grid to get the fine partition
    refined_partition = split_partition(partition,my_stats["mins"])    
    n_rect = len(refined_partition)
    beta_approx_store = np.zeros((n_rect,dim+1))
    alpha_store_refined = np.zeros((n_rect,))
    beta_approx = np.zeros((dim+1,))
    idx = 0
    for rect in refined_partition:
        
        if verbose and np.mod(idx,100) == 0:
            print("Treating rectangle number {} / {}...".format(idx+1,n_rect))
        
        # first we must get the value on the rectangle: we get it at the middle point
        mid = get_middle_point(rect)
        alpha_store_refined[idx] = tree_regressor.predict(mid.reshape(1, -1))
        
        # then the beta corresponding to the rectangle
        beta_approx_store[idx] = compute_beta_indicator_large_nu(xi,rect,my_stats)
        idx += 1

    # final theoretical results are just the weighted sums
    beta_approx = alpha_store_refined.dot(beta_approx_store)
    
    return beta_approx
