# -*- coding: utf-8 -*-
"""

@author: dgarreau

In this experiment we check our theoretical predictions for a CART tree. This 
is Figure 11 in the paper.

"""

import numpy as np
import matplotlib.pyplot as plt

import lime
import lime.lime_tabular

from sklearn.tree import DecisionTreeRegressor

from utils.plot_functions import plot_whisker_boxes

from utils.aux_functions import get_training_data_stats
from utils.aux_functions import format_coefs
from utils.aux_functions import uniform_sample

from theory.tree import compute_beta_tree

if __name__ == "__main__":
    
    # number of experiments
    n_exp = 100
    
    # number of perturbed samples
    n_samples = 5000
    
    # bandwidth parameter
    nu = 10

    # number of bins along each dimension
    p = 4
    
    # for reproducibility
    np.random.seed(3)

    # dimension of the ambient space
    dim = 10
    
    # we work on the [-10,10]^dim square
    theo_boundaries = np.zeros((dim,2))
    theo_boundaries[:,0] = -10*np.ones((dim,))
    theo_boundaries[:,1] = 10*np.ones((dim,))

    # function to regress is the sum of coordinates
    def my_function(data):   
        return np.sum(data,1)

    # training set
    n_train = 1000
    train_data = uniform_sample(theo_boundaries,n_train)
    y_train = my_function(train_data)
    
    # getting summary statistics of the train data
    my_stats = get_training_data_stats(train_data,p)
    
    # creating the tree regressor
    depth = 3
    tree_regressor = DecisionTreeRegressor(random_state=0,max_depth=depth)
    
    # let us train it
    tree_regressor.fit(train_data,y_train)

    # example to explain
    xi = np.random.uniform(-10,10,(dim,))
    
    # let us get the empirical explanations
    explainer = lime.lime_tabular.LimeTabularExplainer(train_data, 
                                                       mode='regression',
                                                       feature_selection='none',
                                                       training_data_stats=my_stats,
                                                       kernel_width=nu)
    
    # get the empirical explanations
    beta_emp_store = np.zeros((n_exp,dim+1))
    for i_exp in range(n_exp):
        
        if np.mod(i_exp + 1,10) == 0:
            s_exp = "Experiment {} / {} is running...".format(i_exp + 1,n_exp)
            print(s_exp)
        
        exp = explainer.explain_instance(xi,tree_regressor.predict,num_samples=n_samples)
        beta_emp_store[i_exp,:] = format_coefs(exp)

    # get the theory 
    # this can take a while if dim or depth are not small
    print("Computing the theory...")
    beta_theo = compute_beta_tree(xi,nu,tree_regressor,my_stats)

#    #################################################################


    # plot empirical vs theory
    fig, ax = plt.subplots(figsize=(15,10))
    plot_whisker_boxes(beta_emp_store,ax,theo=beta_theo,title=r"Coefficients of the surrogate model for a CART tree")

    s_name = "results/tree_explanation"
    fig.savefig(s_name + '.pdf',format='pdf',bbox_inches = 'tight',pad_inches = 0) 
    








