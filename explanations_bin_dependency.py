# -*- coding: utf-8 -*-
"""

@author: dgarreau

Explanations only depend on the bin. In this experiment, we sample points in 
the same d-dimensional bin and compare the explanations given by LIME. This 
script produces Figure 6 in the paper.

"""


import numpy as np
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process.kernels import RBF

from utils.aux_functions import get_training_data_stats
from utils.plot_functions import plot_whisker_boxes

from theory.general import get_bxi

if __name__ == "__main__":

    # number of experiments    
    n_exp = 100
    
    # number of perturbed samples
    n_samples = 1000
    
    # for reproducibility
    np.random.seed(1)

    # loading data
    wine = load_wine()
    X_orig = wine['data']
    Y = wine['target']

    # scaling the data
    X_scaled = preprocessing.scale(X_orig)
    
    # train test split
    X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size = 0.2, random_state=5)
    
    # let us train a kernel regressor
    kernel = RBF(length_scale=10)
    krr_model = KernelRidge(kernel=kernel,alpha=1)
    krr_model.fit(X_train,Y_train)
    def my_model(array):
        return krr_model.predict(array)
    
    # dimension of the ambient space
    dim = X_scaled.shape[1]

    # the example to explain
    xi = X_scaled[0,:]

    my_features = ['alcohol','Malic acid','ash','alcalinity','Mg',
                       'phenols','flavanoids','non-flavanoid phenol',
                       'proanthocyanins','color intensity','hue',
                       'part dil. wines','proline']

    # bandwidth parameter
    nu = 5
    
    # number of bins along each dimension
    p = 4

    # getting the summary statistics
    my_stats = get_training_data_stats(X_scaled,p)

    # creating the explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(X_scaled, 
                                                       mode='regression',
                                                       feature_names=my_features,
                                                       feature_selection='none',
                                                       training_data_stats=my_stats,
                                                       kernel_width=nu)

    # getting the feature names
    exp = explainer.explain_instance(xi, 
                                     my_model, 
                                     num_samples=n_samples)
    names = [x[0] for x in exp.as_list()]    
    
    # getting the bins indices of xi
    bxi = get_bxi(xi,my_stats)
    
    # sampling a second exmaple \xi' into to the same bin
    xi_prime = np.zeros((dim,))
    for j in range(dim):
        left = my_stats["mins"][j][bxi[j]]
        right = my_stats["maxs"][j][bxi[j]]
        xi_prime[j] = np.random.uniform(left,right)

    # getting explanations for xi
    beta_emp_store = np.zeros((n_exp,dim))
    beta_emp_store_prime = np.zeros((n_exp,dim))
    for i_exp in range(n_exp):
        
        if np.mod(i_exp + 1,10) == 0:
            s_exp = "Experiment {} / {} is running...".format(i_exp + 1,n_exp)
            print(s_exp)
        
        exp = explainer.explain_instance(xi, 
                                         my_model, 
                                         num_samples=n_samples)
        
        # getting the right feature names
        for x in exp.as_list():
            for j,name in zip(range(dim),names):
                if x[0] == name:
                    beta_emp_store[i_exp,j] = x[1]

        # same idea for xi'
        exp = explainer.explain_instance(xi_prime, 
                                         my_model, 
                                         num_samples=n_samples)
        for x in exp.as_list():
            for j,name in zip(range(dim),names):
                if x[0] == name:
                    beta_emp_store_prime[i_exp,j] = x[1]

    ###########################################################################
    # plotting the results
    
    fig, ax = plt.subplots(figsize=(15,10))
    plot_whisker_boxes(beta_emp_store,
                       ax,
                       title=r"Local explanation for $\xi$",
                       rotate=True,
                       feature_names=names)

    s_name = "results/explanations_bin_1"
    fig.savefig(s_name + '.pdf',format='pdf',bbox_inches = 'tight',pad_inches = 0) 

    fig1, ax1 = plt.subplots(figsize=(15,10))
    plot_whisker_boxes(beta_emp_store_prime,
                       ax1,
                       title=r"Local explanation for $\xi'$",
                       rotate=True,
                       feature_names=[])
        
    s_name = "results/explanations_bin_2"
    fig1.savefig(s_name + '.pdf',format='pdf',bbox_inches = 'tight',pad_inches = 0) 
 




