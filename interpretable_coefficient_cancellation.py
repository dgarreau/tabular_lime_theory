# -*- coding: utf-8 -*-
"""

@author: dgarreau

For certain choices of hyperparameters, the interpretable coefficients can 
vanish, thus effectively disappearing from the explanation. We demonstrate 
this effect for a linear function with uniform training data. Changing the 
number of bins from 4 (the default) to 5 puts nearly all interpretable 
coefficientsto 0, both in theory and in practice. This is Figure 9 in the 
paper.

"""

import numpy as np
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt

from utils.aux_functions import get_training_data_stats
from utils.aux_functions import format_coefs
from utils.plot_functions import plot_whisker_boxes

from theory.linear import compute_beta_linear


if __name__ == "__main__":
    
    # number of experiments
    n_exp = 100
    
    # number of perturbed samples
    n_samples = 5000
    
    # for reproducibility
    np.random.seed(1)

    # dimension of the ambient space
    dim = 10
    
    # we look into a linear model
    f = np.random.uniform(0,10,(dim+1,))

    def my_model(array):
        return f[0] + np.dot(array,f[1:])

    # the example to explain
    xi = np.zeros((dim,))

    # training set
    n_train = 1000
    train = np.random.uniform(-10,10,(n_train,dim))

    # main loop
    for p in [4,5]:
        print("Number of bins: {}".format(p))

        # getting the summary statistics
        my_stats = get_training_data_stats(train,p)

        # creating the explainer
        explainer = lime.lime_tabular.LimeTabularExplainer(train, 
                                                           mode='regression',
                                                           feature_selection='none',
                                                           training_data_stats=my_stats)

        # we store the empirical values of the explanations in this array
        beta_emp_store = np.zeros((n_exp,dim+1))
        for i_exp in range(n_exp):
            
            if np.mod(i_exp + 1,10) == 0:
                s_exp = "Experiment {} / {} is running...".format(i_exp + 1,n_exp)
                print(s_exp)
            
            # getting the explanation
            exp = explainer.explain_instance(xi, my_model, 
                                         num_samples=n_samples)

            # getting the coefficients of the surrogate model
            beta_emp_store[i_exp,:] = format_coefs(exp)

    

        # getting the values given by theory
        default_nu = np.sqrt(0.75*dim)
        beta_theo = compute_beta_linear(xi,f,default_nu,my_stats)


        ###################################################################

        fig, ax = plt.subplots(figsize=(15,10))
        plot_whisker_boxes(beta_emp_store,
                       ax,
                       theo=beta_theo,
                       title="Coefficients of surrogate model, $p={}$".format(p))

        s_name = "results/cancellation_{}_boxes".format(p)
        fig.savefig(s_name + '.pdf',format='pdf',bbox_inches = 'tight',pad_inches = 0) 




    



