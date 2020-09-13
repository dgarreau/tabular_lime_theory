# -*- coding: utf-8 -*-
"""

@author: dgarreau

Theory vs practice for an indicator function. In this script, we confront 
Theorem 1 (more precisely Proposition 6) to empirical observations. The result 
is Figure 16 in the paper.

"""




import numpy as np
import matplotlib.pyplot as plt

import lime
import lime.lime_tabular

from utils.aux_functions import get_training_data_stats
from utils.aux_functions import format_coefs
from utils.plot_functions import plot_whisker_boxes

from theory.indicator import compute_beta_indicator

if __name__ == "__main__":
    
    # number of experiments
    n_exp = 100
    
    # number of samples (here we need a lot since f is mostly flat)
    n_samples = 10000
    
    # for reproducibility
    np.random.seed(1)

    # dimension of the ambient space
    dim = 5

    # our model is an indicator function with rectangular support
    rect = np.zeros((dim,2))
    rect[:,0] = 2.5*np.ones((dim,))
    rect[:,1] = 7.5*np.ones((dim,))
    def my_model(array):
        n_sample,dim = array.shape
        res = np.zeros((n_sample,))
        for i in range(n_sample):
            x = array[i,:]
            if np.all(x > rect[:,0]) and np.all(x < rect[:,1]):
                res[i] = 1
                
        return res

    # the example to explain
    xi = np.random.uniform(-10,10,(dim,))

    # training set
    n_train = 1000
    train = np.random.uniform(-10,10,(n_train,dim))

    # bandwidth parameter
    nu = 10
    
    # number of bins along each dimension
    p = 10

    # creating the discretizer
    my_stats = get_training_data_stats(train,p)

    # creating the explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(train, 
                                                       mode='regression',
                                                       feature_selection='none',
                                                       training_data_stats=my_stats,
                                                       kernel_width=nu)

    beta_emp_store = np.zeros((n_exp,dim+1))
    for i_exp in range(n_exp):
        
        if np.mod(i_exp + 1,10) == 0:
            s_exp = "Experiment {} / {} is running...".format(i_exp + 1,n_exp)
            print(s_exp)
        
        # getting the explanation
        exp = explainer.explain_instance(xi, 
                                         my_model, 
                                         num_samples=n_samples)

        # getting the coefficients of the local model
        beta_emp_store[i_exp,:] = format_coefs(exp)

    # computing the theoretical values
    beta_theo = compute_beta_indicator(xi,rect,nu,my_stats)

    ###########################################################################
    
    # plotting the results
    fig, ax = plt.subplots(figsize=(15,10))
    plot_whisker_boxes(beta_emp_store,
                           ax,
                           theo=beta_theo)
        
    # saving the fig
    s_name = "results/indicator_explanation"
    fig.savefig(s_name + '.pdf',format='pdf',bbox_inches = 'tight',pad_inches = 0) 





