# -*- coding: utf-8 -*-
"""

@author: dgarreau

In this script, we show how Tabular LIME ignores unused coordinates for a 
non-linear f. We take the example of a kernel ridge regressor trained on a 
subset of the coordinates of the Wine dataset. This is Figure 15 in the paper.

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

from utils.aux_functions import get_training_data_stats
from utils.aux_functions import format_coefs

from utils.plot_functions import plot_whisker_boxes

if __name__ == "__main__":
    
    # number of experiments
    n_exp = 100
    
    # number of perturbed samples
    n_samples = 1000
    
    # for reproducibility
    np.random.seed(1)

    # non linear model, train some kernel ridge regressor on Wine
    wine = load_wine()
    X_orig = wine['data']
    
    # removing the last n_removed coordinate
    n_removed = 6
    X = X_orig[:,:-n_removed]
    Y = wine['target']
    
    # scale evrything
    X_scaled = preprocessing.scale(X)
    X_orig_scaled = preprocessing.scale(X_orig)
    
    # train test split
    X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size = 0.2, random_state=5)
    
    # let us train the model
    kernel = RBF(length_scale=10)
    krr_model = KernelRidge(kernel=kernel,alpha=1)
    krr_model.fit(X_train,Y_train)

    # and restrict to the first coordinates
    def my_model(array):
        return krr_model.predict(array[:,:-n_removed])
    
    # dimension of the ambient space
    dim = X_orig.shape[1]

    # the example to explain
    xi = X_orig_scaled[0,:]

    # bandwidth parameter
    nu = 5
    
    # number of bins along each dimension
    p = 4

    # getting the stats
    my_stats = get_training_data_stats(X_orig_scaled,p)

    # creating the explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(X_orig_scaled, 
                                                       mode='regression',
                                                       feature_selection='none',
                                                       training_data_stats=my_stats,
                                                       kernel_width=nu)

    # main loop
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

    ###########################################################################

    # getting nicer feature names
    #my_features = wine['feature_names']
    my_features = ['alcohol','Malic acid','ash','alcalinity','Mg',
                      'phenols','flavanoids','non-flavanoid phenol',
                      'proanthocyanins','color intensity','hue',
                      'part of diluted wines','proline']


    # plotting the result
    fig, ax = plt.subplots(figsize=(15,10))
    plot_whisker_boxes(beta_emp_store,
                   ax,
                   rotate=True,
                   feature_names=my_features)

    s_name = "results/ignore_non_linear_default_weights"
    fig.savefig(s_name + '.pdf',format='pdf',bbox_inches = 'tight',pad_inches = 0) 

    



