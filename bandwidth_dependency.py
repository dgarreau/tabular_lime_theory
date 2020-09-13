# -*- coding: utf-8 -*-
"""

@author: dgarreau

In this experiments, we look into the behavior of interpretable coefficients 
when the bandwidth parameter varies. The model is a kernel regressor trained 
on the Wine dataset. This script produces Figure 7 in the paper.

"""

import numpy as np
import matplotlib.pyplot as plt

import lime
import lime.lime_tabular

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process.kernels import RBF

from utils.aux_functions import format_coefs,get_training_data_stats

if __name__ == "__main__":
    
    # number of experiments
    n_exp = 100
    
    # number of perturbed samples
    n_samples = 1000
    
    # number of bins along each dimension
    p = 4
    
    # for reproducibility
    np.random.seed(1)

    # bandwidth parameter for the RBF kernel
    gamma = 5
    
    # get the data and train a kernel ridge regressor
    wine = load_wine()
    X_orig = wine['data']
    Y = wine['target']
    X_scaled = preprocessing.scale(X_orig)
    X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size = 0.2, random_state=5)
    kernel = RBF(length_scale=gamma)
    krr_model = KernelRidge(kernel=kernel,alpha=1)
    krr_model.fit(X_train,Y_train)

    # the model to explain is the kernel ridge regressor we just trained
    def my_model(array):
        return krr_model.predict(array)    
    
    # dimension of the ambient space
    dim = X_scaled.shape[1]

    # the example to explain
    xi = X_scaled[0,:]

    # grid of bandwidths
    n_nu    = 50
    nu_min  = 0.001
    nu_max  = 4
    nu_grid = np.linspace(nu_min,nu_max,num=n_nu)
    
    # get the summary of the training data
    my_stats = get_training_data_stats(X_scaled,p)

    coef_store = np.zeros((n_nu,dim+1))
    std_store = np.zeros((n_nu,dim+1))
    beta_emp_store = np.zeros((n_nu,n_exp,dim+1))
    
    # main loop: for each bandwidth obtain many LIME explanations
    for i_nu in range(n_nu):
        
        nu = nu_grid[i_nu]
        s_nu = "bandwidth: {}".format(nu)
        print(s_nu)

        # creating the explainer
        explainer = lime.lime_tabular.LimeTabularExplainer(X_orig, 
                                                           mode='regression',
                                                           feature_selection='none',
                                                           training_data_stats=my_stats,
                                                           kernel_width=nu)
        
        for i_exp in range(n_exp):
            
            if np.mod(i_exp + 1,10) == 0:
                s_exp = "Experiment {} / {} is running...".format(i_exp + 1,n_exp)
                print(s_exp)
            
            # getting the explanation
            exp = explainer.explain_instance(xi, 
                                             my_model, 
                                             num_samples=n_samples)

            # getting the coefficients of the local model
            beta_emp_store[i_nu,i_exp,:] = format_coefs(exp)

        coef_store[i_nu] = np.mean(beta_emp_store[i_nu,:,:],axis=0)
        std_store[i_nu] = np.std(beta_emp_store[i_nu,:,:],axis=0)


    # default choice of bandwidth
    default_nu = np.sqrt(0.75*dim)

    ###########################################################################

    small_fs = 30
    big_fs = 40
    lw = 3
    ms = 6
        
    for index in range(dim):
        
        plt.rc('xtick',labelsize=small_fs)
        plt.rc('ytick',labelsize=small_fs) 
        fig, ax = plt.subplots(figsize=(15,10))
        s_title = "Coefficient {} as a function of the bandwidth".format(index+1)
        ax.set_title(s_title,fontsize=big_fs)
        ax.plot(nu_grid,coef_store[:,index+1],color='k',linewidth=lw,zorder=5)
        
        # shadede area is standard deviation over the experiments
        ax.fill_between(nu_grid,coef_store[:,index+1]-std_store[:,index+1],coef_store[:,index+1]+std_store[:,index+1],color='b',alpha=0.5)
        ax.axhline(y=0,c='k',linestyle='--')
        ax.axvline(x=default_nu,c='r',linewidth=lw)
        ax.set_xlabel("bandwidth",fontsize=small_fs)
        
        # saving the figure
        s_name = "results/coef_vs_bandwidth_wine_krr_{}".format(index)
        fig.savefig(s_name + '.pdf',format='pdf',bbox_inches = 'tight',pad_inches = 0) 


        


