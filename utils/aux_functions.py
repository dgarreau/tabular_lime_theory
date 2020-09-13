# -*- coding: utf-8 -*-
"""

@author: dgarreau

All auxilliary functions are collected in this file.

"""

import numpy as np

def uniform_sample(rect,n_sample):
    """
    Uniformly sampling data on a rectangle.
    
    INPUT:
        rect: rectangle (size (dim,2))
        n_sample: number of points to samples
        
    OUTPUT:
        sample: size (n_sample,dim)
        
    """
    dim = rect.shape[0]
    sample = np.zeros((n_sample,dim))
    
    for j in range(dim):
        sample[:,j] = np.random.uniform(rect[j,0],rect[j,1],(n_sample,))

    return sample

def get_middle_point(rect):
    """
    Gets the middle point of a rectangle.
    
    INPUT:
        rect: hyper-rectangle (size (d,2))
        
    OUTPUT:
        middle point (size (d,))
        
    """
    return 0.5*(rect[:,0] + rect[:,1])

def get_training_data_stats(train,p):
    """
    This function computes training data summaries in the same way Tabular 
    LIME does. 
    
    INPUT:
        train: training data (size (n_train,dim))
        p: number of bins along each dimension
        
    OUTPUT:
        my_stats["means"]: empirical mean on each bin
        my_stats["stds"]: standard deviation on each bin
        my_stats["mins"]: left side of the bin
        my_stats["maxs"]: right side of the bin
        my_stats["feature_values"]: bin indices
        my_stats["feature_frequencies"]: proba to choose the bin (1/p in our case)
        my_stats["bins"]: bins boundaries
        
    """
    n_train,dim = train.shape
    my_stats = {}
    my_stats["means"]               = {}
    my_stats["stds"]                = {}
    my_stats["mins"]                = {}
    my_stats["maxs"]                = {}
    my_stats["feature_values"]      = {}
    my_stats["feature_frequencies"] = {}
    my_stats["bins"] = {}
    
    for j in range(dim):
        data_along_j = train[:,j]
        box_boundaries_along_j = np.percentile(data_along_j, np.arange(p+1)*100/p)
        my_stats["means"][j] = []
        my_stats["stds"][j]  = []
        my_stats["mins"][j]  = []
        my_stats["maxs"][j]  = []
        my_stats["feature_values"][j] = np.arange(p,dtype=float)
        my_stats["feature_frequencies"][j] = list((1/p)*np.ones((p,)))
        my_stats["bins"][j] = box_boundaries_along_j[1:-1]
        for b in range(p):
            left  = box_boundaries_along_j[b]
            right = box_boundaries_along_j[b+1]
            
            select_bool = (left <= data_along_j) * (data_along_j <= right)
            
            selection = data_along_j[select_bool]
            
            my_stats["means"][j].append(np.mean(selection))
            my_stats["stds"][j].append(np.std(selection))
            my_stats["mins"][j].append(left)
            my_stats["maxs"][j].append(right)
    
    return my_stats

def format_coefs(explanation):
    """
    Formats the explanations provided by Tabular LIME in a convenient way.
    
    INPUT:
        explanation: explanation provided by LIME (lime.explanation.Explanation)
        
    OUTPUT:
        betahat: interpretable coefficients, with intercept first position (size (dim+1,))
    
    """
    intercept = explanation.intercept
    coefs = explanation.local_exp
    dim = len(coefs[0])
    betahat = np.zeros((dim+1,))
    betahat[0] = intercept[0]
    for i in range(dim):
        aux = [item for item in coefs[0] if item[0] == i]
        if len(aux) == 1:
            betahat[i+1] = -aux[0][1]
    return betahat

def get_grid_params(my_stats):
    """
    Get bins middle points and boundaries of the LIME partition when plotting 
    explanations in a 2D grid.
    
    INPUT:
        my_stats: summary statistics (see get_training_data_stats)
        
    OUTPUT:
        bins_middle_points (size (p,p,2))
        bins_left_boundaries (size (p,p,2))
        bins_right_boundaries (size (p,p,2))
        
    """
    dim = len(my_stats["means"])
    if dim != 2:
        raise ValueError("Data should be 2-dimensional.")
    p = len(my_stats["means"][0])
    bins_middle_points = np.zeros((p,p,2))
    bins_left_boundaries = np.zeros((p,p,2))
    bins_right_boundaries = np.zeros((p,p,2))
    for i in range(p):
        for j in range(p):
            l1 = my_stats["mins"][0][i]
            r1 = my_stats["maxs"][0][i]
            l2 = my_stats["mins"][1][j]
            r2 = my_stats["maxs"][1][j]
            bins_left_boundaries[i,j,0] = l1
            bins_left_boundaries[i,j,1] = l2
            bins_right_boundaries[i,j,0] = r1
            bins_right_boundaries[i,j,1] = r2
            bins_middle_points[i,j,0] = 0.5*(l1+r1)
            bins_middle_points[i,j,1] = 0.5*(l2+r2)
    return bins_middle_points,bins_left_boundaries,bins_right_boundaries

def get_partition(regressor,domain_boundaries):
    """
    This function retrieves the partition of the input space given by a tree 
    regressor.
    
    INPUT:
        regressor: tree regressor from sklearn
        domain_boundaries: limits of the input space (size (dim,2))
        
    OUTPUT:
        partition: a list of rectangles, each of size (dim,2)
    
    """
    depth = regressor.get_depth()
    children_left = regressor.tree_.children_left
    children_right = regressor.tree_.children_right
    feature = regressor.tree_.feature
    threshold = regressor.tree_.threshold

    # get the node indices of the leaves
    leaves = np.where(feature == -2)[0]
    n_leaves = leaves.shape[0]

    # for each leaf, get the path to the leaf
    paths = []
    direcs = []

    for leaf in leaves:

        current = leaf
        path_list = [current]
        direc = []
        for d in range(depth):
            aux_left = np.where(children_left == current)[0]
            aux_right = np.where(children_right == current)[0]
            if len(aux_left) != 0:
                direc.append(False)
                current = aux_left[0]
            if len(aux_right) != 0:
                direc.append(True)
                current = aux_right[0]
            path_list.append(current)
            if current == 0:
                break

        paths.append(path_list)
        direcs.append(direc)

    # then get the limits
    partition = []
    for idx in range(n_leaves):

        current_path = paths[idx]
        current_depth = len(current_path)
        current_direc = direcs[idx]

        # start with the domain boundaries
        current_rect = np.copy(domain_boundaries)
        
        # then at each level we cut according to the node
        for d in range(current_depth-1):

            current_node = current_path[current_depth-d-1]
            feat = feature[current_node]
            thresh = threshold[current_node]
            
            # if True, the arrow points right and we split for greater values
            if current_direc[current_depth-d-2]:
                current_rect[feat,0] = thresh
            else:
                current_rect[feat,1] = thresh
        
        # add the rectangle to the partition
        partition.append(current_rect)
    return partition

def split_rectangle(rect,d,values):
    """
    This function splits a rectangle along a given direction at the given 
    values.
    
    INPUT:
        rect: rectangle to split (size (dim,2))
        d: direction in which to split
        values: list of values
        
    OUTPUT:
        a list of rectangles

    """
    # sort the values in increasing order
    sorted_values = np.sort(values)
    
    # the result is a list of rectangles
    out = []
    current_left = rect[d,0]
    
    # we browse the values
    for v in sorted_values:

        # if they cut the rectangle along dimension d
        if current_left < v and rect[d,1] > v:

            # then we create a new rectangle to the left
            new_rect = np.copy(rect)
            new_rect[d,0] = current_left
            new_rect[d,1] = v
            current_left = v
            out.append(new_rect)
    
    # and the last rectangle in any case (if not splitted, original rectangle)
    new_rect = np.copy(rect)
    new_rect[d,0] = current_left
    out.append(new_rect)
        
    return out

def split_partition_aux(partition,d,values):
    """
    Split a partition along a certain direction for given values. 
    
    INPUT:
        partition: a list of rectangles of size (dim,2)
        d: direction in which to split
        values: list of values
        
    OUTPUT:
        list of rectangles of size (dim,2)
        
    """
    out = []
    for rect in partition:
        splitted_rect = split_rectangle(rect,d,values)
        out += splitted_rect
    return out

def split_partition(partition,values_dict):
    """
    Splitting a partition according to a grid of values.
    
    INPUT: 
        partition: list of rectangles of size (dim,2)
        values_dict: values[d] is a list of values
        
    OUTPUT: 
        list of rectangles of size (dim,2)
        
    """
    out = partition.copy()
    dim = len(values_dict)
    for d in range(dim):
        out = split_partition_aux(out,d,values_dict[d])
    return out

def get_empirical_boundaries(my_stats):
    """
    Get the empirical boundaries from summary statistics.
    
    INPUT:
        my_stats: summary_statistics (see get_training_data_stats)
        
    OUTPUT:
        boundaries
        
    """
    dim = len(my_stats["means"])
    emp_boundaries = np.zeros((dim,2))
    emp_boundaries[:,0] = [my_stats["mins"][x][0] for x in my_stats["mins"]]
    emp_boundaries[:,1] = [my_stats["maxs"][x][-1] for x in my_stats["maxs"]]
    return emp_boundaries

