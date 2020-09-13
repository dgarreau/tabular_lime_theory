# -*- coding: utf-8 -*-
"""

@author: dgarreau

Investigating the effects of regularization on the surrogate model. For three 
different regularization parameters, we plot the empirical explanations and the 
theory for a linear function. This script produces Figure 3 of the paper. 

"""

import numpy as np

import matplotlib.pyplot as plt

import lime
import lime.lime_tabular

from sklearn.linear_model import Ridge

from utils.plot_functions import plot_whisker_boxes

from utils.aux_functions import format_coefs
from utils.aux_functions import get_training_data_stats

from theory.linear import compute_beta_linear

if __name__ == "__main__":
    
    # for reproducibility
    np.random.seed(1)
    
    # number of experiments
    n_exp = 100
    
    # number of perturbed samples (5k is default)
    n_samples = 5000

    # dimension of the ambient space
    dim = 10
    
    # f is a linear function
    f = np.random.uniform(0,10,(dim+1,))
    def my_model(array):
        return f[0] + np.dot(array,f[1:])

    # training set
    n_train = 1000
    train = np.random.normal(0,1,(n_train,dim))

    # bandwidth parameter for the explanation
    nu = 10
    
    # number of bins along each dimension
    p = 4
    
    # getting summary statistics for the train
    my_stats = get_training_data_stats(train,p)
    
    # choosing the example to explain
    xi = np.zeros((dim,))

    # creating the explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(train, 
                                                       mode='regression',
                                                       feature_selection='none',
                                                       training_data_stats=my_stats,
                                                       kernel_width=nu)
    # values of the regularization parameter
    lbda_grid = [0,1,1000]
    
    ylims = [-4,4]
    
    for lbda in lbda_grid:
    
        print("Regularization parameter: lambda = {}".format(lbda))
        
        # surrogate model obtained by ridge with regularization constant lbda
        ridge_regressor = Ridge(alpha=lbda, fit_intercept=True)

        # empirical explanations are stored in this arry
        beta_emp_store = np.zeros((n_exp,dim+1))
        
        # a few repetitions to take randomness of sampling into account
        for i_exp in range(n_exp):
            
            if np.mod(i_exp + 1,10) == 0:
                s_exp = "Experiment {} / {} is running...".format(i_exp + 1,n_exp)
                print(s_exp)
            
            # getting the explanation at xi
            exp = explainer.explain_instance(xi, 
                                             my_model, 
                                             model_regressor=ridge_regressor, 
                                             num_samples=n_samples)
            beta_emp_store[i_exp,:] = format_coefs(exp)

        # getting the theoretical values
        beta_theo = compute_beta_linear(xi,f,nu,my_stats)

###############################################################################

        # plotting the result
        fig, ax = plt.subplots(figsize=(15,10))
        plot_whisker_boxes(beta_emp_store,
                           ax,
                           title="Coefficients of the surrogate model ($\lambda={}$)".format(lbda),
                           theo=beta_theo,
                           ylims=ylims)
        
        # and save it
        s_name = "results/regularization_default_linear_lbda_{}".format(lbda)
        fig.savefig(s_name + '.pdf',format='pdf',bbox_inches = 'tight',pad_inches = 0) 





