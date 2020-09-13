# -*- coding: utf-8 -*-
"""

@author: dgarreau

Looking at explanations for an indicator function with rectangular support in 
dimension 2. Since the explanations do not depend on the exact value of xi, 
we plot one explanation per 2-dimensional bin at the center. This is Figure 10 
in the paper.

"""


import numpy as np
import matplotlib.pyplot as plt

import lime
import lime.lime_tabular

from utils.aux_functions import get_training_data_stats
from utils.aux_functions import get_grid_params
from utils.aux_functions import format_coefs

from utils.plot_functions import plot_grid

from theory.indicator import compute_beta_indicator

if __name__ == "__main__":
    
    # number of experiments
    n_exp = 10
    
    # number of perturbed samples
    n_samples = 1000
    
    # for reproducibility
    np.random.seed(3)

    # we work in 2D
    dim = 2

    # the model to explain is an indicator function with rectangular support
    rect = np.array([[-4,0],[-9,-6]])
    def my_model(array):
        n_sample,dim = array.shape
        res = np.zeros((n_sample,))
        for i in range(n_sample):
            x = array[i]
            if x[0] > rect[0,0] and x[0] < rect[0,1] and x[1] > rect[1,0] and x[1] < rect[1,1]:
                res[i] = 1
        return res

    # uniform training data
    x_min = -10
    x_max = 10
    y_min = x_min
    y_max = x_max
    n_train = 1000
    train = np.random.uniform(x_min,x_max,(n_train,dim))

    # bandwidth parameter
    nu = 10
    
    # number of bins along each dimension
    p = 4

    # getting summary statistics of the train data
    my_stats = get_training_data_stats(train,p)
    
    # getting the parameters of the grid            
    bins_middle_points,bins_left_boundaries,bins_right_boundaries = get_grid_params(my_stats)
    
    # creating the explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(train, 
                                                       mode='regression',
                                                       feature_selection='none',
                                                       training_data_stats=my_stats,
                                                       kernel_width=nu)

    # we will store the empirical explanations in this array
    beta_emp_store = np.zeros((p,p,n_exp,dim+1))
    
    # main loop
    for i_exp in range(n_exp):
        
        if np.mod(i_exp + 1,1) == 0:
            s_exp = "Experiment {} / {} is running...".format(i_exp + 1,n_exp)
            print(s_exp)
        
        # traversing the grid
        for i in range(p):
            for j in range(p):
                
                # taking xi in the middle of the box 
                xi = np.array([bins_middle_points[i,j,0],bins_middle_points[i,j,1]])
                
                # getting the explanation
                exp = explainer.explain_instance(xi, 
                                                 my_model, 
                                                 num_samples=n_samples)

                # getting the coefficients of the local model
                beta_emp_store[i,j,i_exp,:] = format_coefs(exp)
                

    # averaging over all experiments
    beta_mean = np.mean(beta_emp_store[:,:,:,1:],2)

    # looking at the norm can help to fix the size of the arrows
    beta_norm = np.zeros((p,p))
    for i in range(p):
        for j in range(p):
            beta_norm[i,j] = np.linalg.norm(beta_mean[i,j,:])

    # theoretical values
    beta_theo = np.zeros((p,p,2))
    for i in range(p):
        for j in range(p):
            xi = np.array([bins_middle_points[i,j,0],bins_middle_points[i,j,1]])
            beta_theo[i,j,:] = compute_beta_indicator(xi,rect,nu,my_stats)[1:]
           
    ###########################################################################

    center = rect
        
    # first figure: empirical
    fig, ax = plt.subplots(figsize=(10,10))
    plot_grid(beta_mean,
              ax,
              my_stats,
              center,
              scale=25)


        
    # saving the fig
    s_name = "results/two-dimensional_tree_vector_field_empirical"
    fig.savefig(s_name + '.pdf',format='pdf',bbox_inches = 'tight',pad_inches = 0)  
 
    
    
        
    # second figure: theory
    fig, ax = plt.subplots(figsize=(10,10))
    plot_grid(beta_theo,
              ax,
              my_stats,
              center,
              title="Explanations for an indicator function",
              scale=25)
        
    # saving the fig
    s_name = "results/2D_indicator"
    fig.savefig(s_name + '.pdf',format='pdf',bbox_inches = 'tight',pad_inches = 0)  
            