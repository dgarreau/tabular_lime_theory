# -*- coding: utf-8 -*-
"""

@author: dgarreau

Looking at explanations for a RBF kernel function in dimension 2. Since the 
explanations do not depend on the exact value of xi, we plot one explanation 
per 2-dimensional bin at the center. This is Figure 14 in the paper.

"""

import numpy as np
import matplotlib.pyplot as plt

import lime
import lime.lime_tabular

from utils.aux_functions import get_training_data_stats
from utils.aux_functions import get_grid_params
from utils.aux_functions import format_coefs

from utils.plot_functions import plot_grid

from theory.kernel import compute_beta_gk

if __name__ == "__main__":
    
    # number of experiments
    n_exp = 10
    
    # number of perturbed samples
    n_samples = 1000
    
    # for reproducibility
    np.random.seed(3)

    # dimension of the ambient space
    dim = 2

    # our model is a kernel function
    zeta = np.array([-2.5,-2.5])
    gamma = 1
    def my_model(array):
        return np.exp(-np.sum(np.square(array-zeta),1)/(2*gamma**2))

    x_min = -10
    x_max = 10
    y_min = x_min
    y_max = x_max

            
    # training set
    n_train = 1000
    train = np.random.uniform(x_min,x_max,(n_train,dim))

    # bandwidth parameter
    nu = 1
    
    # number of bins along each dimension
    p = 10

    # getting the data summary
    my_stats = get_training_data_stats(train,p)
    
    # getting the parameters of the grid            
    bins_middle_points,bins_left_boundaries,bins_right_boundaries = get_grid_params(my_stats)

            
    # creating the explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(train, 
                                                       mode='regression',
                                                       feature_selection='none',
                                                       training_data_stats=my_stats,
                                                       kernel_width=nu)

    # we will store the empirical result in this array
    beta_emp_store = np.zeros((p,p,n_exp,dim+1))
    
    # main loop
    for i_exp in range(n_exp):
        
        s_exp = "Experiment number {} running...".format(i_exp)
        print(s_exp)
        
        for i in range(p):
            for j in range(p):
                
                # take xi in the middle of the box 
                xi = bins_middle_points[i,j,:]
                
                # getting the explanation
                exp = explainer.explain_instance(xi, 
                                                 my_model,
                                                 num_samples=n_samples)

                # getting the coefficients of the local model
                beta_emp_store[i,j,i_exp,:] = format_coefs(exp)
                              
    # averaging over all experiments
    beta_mean = np.mean(beta_emp_store[:,:,:,1:],2)

    # theoretical values
    beta_theo = np.zeros((p,p,2))
    for i in range(p):
        for j in range(p):
            xi = np.array([bins_middle_points[i,j,0],bins_middle_points[i,j,1]])
            beta_theo[i,j,:] = compute_beta_gk(xi,nu,zeta,gamma,my_stats)[1:]
    
#    # looking at the norm can be usefule to determine the size of the arrows
#    beta_norm = np.zeros((p,p))
#    for i in range(p):
#        for j in range(p):
#            beta_norm[i,j] = np.linalg.norm(beta_mean[i,j,:])

    ###########################################################################
    
    # first figure: empirical
    fig, ax = plt.subplots(figsize=(10,10))
    plot_grid(beta_mean,
              ax,
              my_stats,
              zeta,
              scale=100)
        
    # now the theory
    fig, ax = plt.subplots(figsize=(10,10))
    plot_grid(beta_theo,
              ax,
              my_stats,
              zeta,
              scale=100)
        
    # saving the fig
    s_name = "results/two-dimensional_kernel_vector_field"
    fig.savefig(s_name + '.pdf',format='pdf',bbox_inches = 'tight',pad_inches = 0)  
            