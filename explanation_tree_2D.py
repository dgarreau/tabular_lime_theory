# -*- coding: utf-8 -*-
"""

@author: dgarreau

In this script, we look into the LIME explanations for a CART tree in 2D. This 
CART tree is trained on a bump in the upper left corner of the [-10,10]^2 
square. As predicted by the theory, all the xi that are aligned with the bump 
along a given coordinate receive a positive explanation, whereas the function 
to explain is very flat in the vidinity of xi. This is Figure 12 in the paper. 

"""

import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt

import lime
import lime.lime_tabular

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics.pairwise import rbf_kernel

import matplotlib.cm as cm

from utils.aux_functions import get_training_data_stats
from utils.aux_functions import get_partition
from utils.aux_functions import format_coefs
from utils.aux_functions import uniform_sample
from utils.aux_functions import get_empirical_boundaries

from utils.plot_functions import plot_partition,plot_bins,plot_whisker_boxes

from theory.tree import compute_beta_tree

if __name__ == "__main__":
 
    # number of experiments
    n_exp = 100
    
    # number of perturbed samples
    n_samples = 1000
    
    # bandwidth parameter
    nu = 1

    # number of bins along each dimension
    p = 4
    
    # for reproducibility
    np.random.seed(3)

    # dimension of the ambient space
    dim = 2
    
    # we work on the [-10,10]^2 square
    theo_boundaries = np.zeros((dim,2))
    theo_boundaries[:,0] = -10*np.ones((dim,))
    theo_boundaries[:,1] = 10*np.ones((dim,))

    # function to regress is a bump in the upper-left corner
    zeta = np.array([-3,3])
    def goal_function(data):
        return rbf_kernel(data,zeta.reshape(1, -1))

    # training set
    n_train = 1000
    train_data = uniform_sample(theo_boundaries,n_train)
    
    # getting summary statistics of the train data
    my_stats = get_training_data_stats(train_data,p)

    # creating the tree regressor
    depth = 4
    tree_reg = DecisionTreeRegressor(random_state=0,max_depth=depth)
    
    # train to fit the goal function
    tree_reg.fit(train_data,goal_function(train_data))

    # examples to explain 
    xi_1 = np.array([-3,2.3])
    xi_2 = np.array([-3,-2.3])
    xi_3 = np.array([-3,-7.5])
    xi_list = [xi_1,xi_2,xi_3]
    n_xi = len(xi_list)
    
    # let us get the empirical explanations
    explainer = lime.lime_tabular.LimeTabularExplainer(train_data, 
                                                       mode='regression',
                                                       feature_selection='none',
                                                       training_data_stats=my_stats,
                                                       kernel_width=nu)
    
    # main loop
    beta_emp_store = np.zeros((n_exp,dim+1,n_xi))
    for i_exp in range(n_exp):
        
        if np.mod(i_exp + 1,10) == 0:
            s_exp = "Experiment {} / {} is running...".format(i_exp + 1,n_exp)
            print(s_exp)
        
        for i_xi in range(n_xi):
            xi = xi_list[i_xi]
            exp = explainer.explain_instance(xi,tree_reg.predict,num_samples=n_samples)
            beta_emp_store[i_exp,:,i_xi] = format_coefs(exp)

    
    # theory
    beta_theo = np.zeros((dim+1,n_xi))
    for i_xi in range(n_xi):
        xi = xi_list[i_xi]
        beta_theo[:,i_xi] = compute_beta_tree(xi,nu,tree_reg,my_stats)
        
    #################################################################
    # train data

    # computing the goal function values for the heatmap
    delta = 0.1
    x = y = np.arange(-10.0, 10.0, delta)
    n = x.shape[0]
    values = np.zeros((n,n))
    for i in range(n):
        for j in range(n):        
            values[i,j] = goal_function(np.array([x[j],y[i]]).reshape(1,-1))


    # plot the partition given by the regressor (in red) and the LIME bins (in black)
    emp_boundaries = get_empirical_boundaries(my_stats)
    partition = get_partition(tree_reg,emp_boundaries)
    fig, ax = plt.subplots(figsize=(10,10))
    plot_partition(partition,ax)
    plot_bins(my_stats,ax)
    ax.imshow(values, interpolation='bilinear', cmap=cm.Blues,
               origin='lower', extent=[-10, 10, -10, 10],
               vmax=abs(values).max(), vmin=0)
    for i_xi in range(n_xi):
        xi = xi_list[i_xi]
        ax.scatter(xi[0],xi[1],color='k')
        s_xi = r"$\xi_" + str(i_xi+1) + "$"
        ax.annotate(s_xi,(xi[0]+0.2,xi[1]+0.2),fontsize=20)
    ax.plot()
    s_title = "2-D tree regressor"
    ax.set_title(s_title,fontsize=40)
    ax.tick_params(labelsize=30)
    ax.set_xlabel(r"$x_1$",fontsize=30)
    ax.set_ylabel(r"$x_2$",fontsize=30)
    s_name = "results/general_situation"
    fig.savefig(s_name + '.pdf',format='pdf',bbox_inches = 'tight',pad_inches = 0)

    # plot empirical vs theory for all the xis
    for i_xi in range(n_xi):
        fig, ax = plt.subplots(figsize=(5,10))
        s_title = r"Explanation for $\xi_" + str(i_xi+1) + "$"
        plot_whisker_boxes(beta_emp_store[:,:,i_xi],ax,theo=beta_theo[:,i_xi],title=s_title,ylims=[-0.06,0.15])
        s_name = "results/explanation_xi_{}".format(i_xi+1)
        fig.savefig(s_name + '.pdf',format='pdf',bbox_inches = 'tight',pad_inches = 0)








