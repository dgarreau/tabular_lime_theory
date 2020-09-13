# -*- coding: utf-8 -*-
"""

@author: dgarreau

Demonstrating linearity of Tabular LIME explanations. We produce explanations 
for two different models as well as their sum for a given example xi. This 
script produces Figure 5 in the paper. 
 
"""

import numpy as np
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt

from utils.plot_functions import plot_whisker_boxes
from utils.aux_functions import get_training_data_stats,format_coefs

if __name__ == "__main__":
    
    # for reproducibility
    np.random.seed(3)
    
    # number of experiments
    n_exp = 100
    
    # number of perturbed samples
    n_samples = 1000

    # dimension of the ambient space
    dim = 10

    # two kernel functions
    zeta_1 = np.zeros((dim,))
    zeta_2 = np.ones((dim,))
    gamma_1 = 10
    gamma_2 = gamma_1
    def my_model_1(array):
        return np.exp(-np.sum(np.square(array-zeta_1),1)/(2*gamma_1**2))
    def my_model_2(array):
        return np.exp(-np.sum(np.square(array-zeta_2),1)/(2*gamma_2**2))
    def model_sum(array):
        return my_model_1(array) + my_model_2(array)

    # the example to explain
    xi = np.random.uniform(-10,10,(dim,))

    # training set: uniform on [-10,10]^dim
    n_train = 1000
    train_set = np.random.uniform(-10,10,(n_train,dim))

    # bandwidth parameter
    nu = 10
    
    # number of bins
    p = 4

    # creating the discretizer
    my_stats = get_training_data_stats(train_set,p)

    # creating the explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(train_set, 
                                                       mode='regression',
                                                       feature_selection='none',
                                                       training_data_stats=my_stats,
                                                       kernel_width=nu)

    beta_emp_store_1   = np.zeros((n_exp,dim+1))
    beta_emp_store_2   = np.zeros((n_exp,dim+1))
    beta_emp_store_sum = np.zeros((n_exp,dim+1))
    for i_exp in range(n_exp):
        
        if np.mod(i_exp + 1,10) == 0:
            s_exp = "Experiment {} / {} is running...".format(i_exp + 1,n_exp)
            print(s_exp)
                
        # getting the explanations for both models and the sum
        exp_1 = explainer.explain_instance(xi, 
                                         my_model_1, 
                                         num_samples=n_samples)
        exp_2 = explainer.explain_instance(xi, 
                                         my_model_2, 
                                         num_samples=n_samples)
        exp_sum = explainer.explain_instance(xi, 
                                         model_sum, 
                                         num_samples=n_samples)

        # getting the coefficients of the local models
        beta_emp_store_1[i_exp,:]   = format_coefs(exp_1)
        beta_emp_store_2[i_exp,:]   = format_coefs(exp_2)
        beta_emp_store_sum[i_exp,:] = format_coefs(exp_sum)

        beta_mean_1   = np.mean(beta_emp_store_1,0)
        beta_mean_2   = np.mean(beta_emp_store_2,0)
        beta_mean_sum = np.mean(beta_emp_store_sum,0)
 

    ###########################################################################
        
    # plotting the interpretable coefficients for the three models
    fig_1, ax_1 = plt.subplots(figsize=(15,10))
    plot_whisker_boxes(beta_emp_store_1,ax_1,theo=beta_mean_1,color="k")
        
    fig_2, ax_2 = plt.subplots(figsize=(15,10))
    plot_whisker_boxes(beta_emp_store_2,ax_2,theo=beta_mean_2,color="k")
        
    fig_sum, ax_sum = plt.subplots(figsize=(15,10))
    plot_whisker_boxes(beta_emp_store_sum,ax_sum,theo=beta_mean_sum,color="red")
        
    # saving the fig
    fig_1.savefig("results/linearity_of_explanations_1.pdf",format='pdf',bbox_inches = 'tight',pad_inches = 0) 
    fig_2.savefig("results/linearity_of_explanations_2.pdf",format='pdf',bbox_inches = 'tight',pad_inches = 0) 
    fig_sum.savefig("results/linearity_of_explanations_sum.pdf",format='pdf',bbox_inches = 'tight',pad_inches = 0) 
