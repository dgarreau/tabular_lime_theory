# -*- coding: utf-8 -*-
"""

@author: dgarreau

Theory vs practice for a linear model. In this script, we confront Theorem 1 
(more precisely Corollary 4) to empirical observations. The result is Figure 8 
in the paper.

"""

import numpy as np
import matplotlib.pyplot as plt

import lime
import lime.lime_tabular

from utils.aux_functions import get_training_data_stats
from utils.aux_functions import format_coefs
from utils.plot_functions import plot_whisker_boxes

from theory.linear import compute_beta_linear


if __name__ == "__main__":
    
    # number of experiments
    n_exp = 100
    
    # number of perturbed examples
    n_sample = 1000
    
    # for reproducibility
    np.random.seed(1)

    # dimension of the ambient space
    dim = 10
    
    # let us define a linear model
    f = np.random.uniform(0,10,(dim+1,))
    def my_model(array):
        return f[0] + np.dot(array,f[1:])

    # the example to explain
    xi = np.zeros((dim,))

    # some random train set
    n_train = 100
    train = np.random.normal(0,1,(n_train,dim))

    # bandwidth parameter
    nu = 10
    
    # number of bins along each dimension
    p = 4

    # creating the discretizer
    my_stats = get_training_data_stats(train,p)

    # creating the explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(train, 
                                                       mode='regression',
                                                       feature_selection='none',
                                                       training_data_stats=my_stats,
                                                       kernel_width=nu)

    # we store the empirical values of beta in this array
    beta_emp_store = np.zeros((n_exp,dim+1))
    for i_exp in range(n_exp):
        
        if np.mod(i_exp + 1,10) == 0:
            s_exp = "Experiment {} / {} is running...".format(i_exp + 1,n_exp)
            print(s_exp)
        
        # getting the explanation
        exp = explainer.explain_instance(xi, 
                                         my_model, 
                                         num_samples=n_sample)

        # getting the coefficients of the local model
        beta_emp_store[i_exp,:] = format_coefs(exp)

    # getting the theoretical values
    beta_theo = compute_beta_linear(xi,f,nu,my_stats)

    ###########################################################################

    # plotting the results
    fig, ax = plt.subplots(figsize=(15,10))
    plot_whisker_boxes(beta_emp_store,
                       ax,
                       theo=beta_theo,
                       title=r"Coefficients of the surrogate model for linear $f$")
    
    fig.savefig('results/linear_f_default_weights_ls.pdf',format='pdf',bbox_inches = 'tight',pad_inches = 0) 

    
    
    
    
    
    
    
    
    