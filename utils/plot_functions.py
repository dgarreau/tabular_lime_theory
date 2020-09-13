# -*- coding: utf-8 -*-
"""

@author: dgarreau

All functions used to plot figures.

"""

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import matplotlib.patches as patches

import numpy as np

from utils.aux_functions import get_grid_params

# numerical constants
small_fs = 30
big_fs = 40
lw = 3
ms = 6


def plot_whisker_boxes(my_data,
                       axis,
                       title=None,
                       xlabel=None,
                       theo=None,
                       rotate=False,
                       feature_names=None,
                       ylims=None,
                       color="red"):
    """
    Plots whisker boxes for interpretable coefficients.
    
    INPUT:
        my_data: raw explanations (size (n_exp,dim+1))
        axis: plt axis (type matplotlib.axes._subplots.AxesSubplot)
        title: title of the figure (str)
        xlabel: label for the x axis (str)
        theo: theoretical values marked by crosses on the plot (size (dim+1,))
        rotate: classical view if True (bool)
        feature_names: default is 1,2,...
        ylims: providing ylims if needed
        color: color of the crosses
        
    """
    
    
    # get the dimension of the data
    dim = my_data.shape[1] -1
    
    # horizontal whiskerboxes
    if rotate:
        axis.boxplot(my_data[:,1:],showmeans=False,
                   boxprops= dict(linewidth=lw, color='black'), 
                   whiskerprops=dict(linestyle='-',linewidth=lw, color='black'),
                   medianprops=dict(linestyle='-',linewidth=lw,color='blue'),
                   flierprops=dict(marker='o',markerfacecolor='black',linewidth=lw),
                   capprops=dict(linewidth=lw),
                   vert=False)
        axis.axvline(x=0,c='k',linestyle='--')
        
        
        y_pos = np.arange(dim) + 1
        axis.set_yticks(y_pos)
        if feature_names is None:
            feature_names = np.arange(1,dim+1)
        
        axis.set_yticklabels(feature_names)
        axis.invert_yaxis()
        axis.tick_params(labelsize=small_fs)
        
    # vertical whisker boxes
    else:
        
        # not plotting the intercept
        axis.boxplot(my_data[:,1:],showmeans=False,
                   boxprops= dict(linewidth=lw, color='black'), 
                   whiskerprops=dict(linestyle='-',linewidth=lw, color='black'),
                   medianprops=dict(linestyle='-',linewidth=lw,color='blue'),
                   flierprops=dict(marker='o',markerfacecolor='black',linewidth=lw),
                   capprops=dict(linewidth=lw))
        
        axis.set_ylim(ylims)
        
        # plotting horizontal line to denote 0
        axis.axhline(y=0,c='k',linestyle='--')
        
    
        # plotting the theoretical predictions if any
        if theo is not None:

            for i_feature in range(dim):
                axis.plot(i_feature+1,
                        theo[i_feature+1],
                        'x',
                        markersize=20,
                        markeredgewidth=5,
                        zorder=10,
                        color=color)
        # setting the labels
        if xlabel is None:
            axis.set_xlabel("features id",fontsize=small_fs)
        
        # setting xticks and yticks
        axis.set_xticklabels(np.arange(1,dim+1), rotation=0, fontsize=small_fs)
        axis.tick_params(labelsize=small_fs)
    
    # setting the title
    if title is None:
        title = "Coefficients of the surrogate model"
    axis.set_title(title,fontsize=big_fs)
    return 0

def plot_bins(my_stats,axis):
    """
    Plots the Tabular LIME bins in 2D given summary statistics.
    
    INPUT:
        my_stats: summary statistics (see get_training_data_stats)
        axis: plt axis (type matplotlib.axes._subplots.AxesSubplot)
    
    """
    p = len(my_stats["means"][0])
    _,bins_left_boundaries,bins_right_boundaries = get_grid_params(my_stats)
    
    # plotting the grid
    x_min_fig = np.min(bins_left_boundaries[:,:,0])
    x_max_fig = np.max(bins_right_boundaries[:,:,0])
    y_min_fig = np.min(bins_left_boundaries[:,:,1])
    y_max_fig = np.max(bins_right_boundaries[:,:,1])
    for i in range(p):
            axis.vlines(x=bins_left_boundaries[i,0,0],ymin=y_min_fig,ymax=y_max_fig)
            axis.hlines(y=bins_left_boundaries[0,i,1],xmin=x_min_fig,xmax=x_max_fig)
            axis.vlines(x=bins_right_boundaries[-1,0,0],ymin=y_min_fig,ymax=y_max_fig)
            axis.hlines(y=bins_right_boundaries[0,-1,1],xmin=x_min_fig,xmax=x_max_fig)
    return 0

def plot_grid(my_data,
              axis,
              my_stats,
              center,
              title=None, 
              scale=25):
    """
    Plotting explanations on a 2-dimensional grid as arrows centered on each 
    2-dimensional bin center. 
    
    INPUT:
        my_data: explanations (size (p,p,2))
        axis: plt axis (type matplotlib.axes._subplots.AxesSubplot)
        my_stats: summary statistics (see get_training_data_stats)
        center: area to mark on the figure, can either be a rectangle or a point
        title: title string
        scale: size of the arrows
        
    """
    p = len(my_stats["means"][0])
    bins_middle_points,bins_left_boundaries,bins_right_boundaries = get_grid_params(my_stats)

    # rectangle case
    if len(center.shape) == 2:
        left_boundaries = center[:,0]
        right_boundaries = center[:,1]
        rect = patches.Rectangle((left_boundaries[0],left_boundaries[1]),
                                 right_boundaries[0]-left_boundaries[0],
                                 right_boundaries[1]-left_boundaries[1],
                                 linewidth=lw,
                                 edgecolor='r',
                                 facecolor='none')
        axis.add_patch(rect)
        axis.annotate(r"$A$",
                    (left_boundaries[0]+0.1,left_boundaries[1]+0.1),
                    color="r",
                    fontsize=small_fs)
        
    # kernel case, we mark a point
    else: 
        axis.scatter(center[0],center[1],c='r',s=2*ms)
        axis.annotate(r"$\zeta$",(center[0]-0.1,center[1]+0.2),color='r',fontsize=small_fs)

    # plotting the vectors
    for i in range(p):
        for j in range(p):
            x_middle = bins_middle_points[i,j,0]
            y_middle = bins_middle_points[i,j,1]
            dx = my_data[i,j,0]
            dy = my_data[i,j,1]
            axis.scatter(x_middle,y_middle,c='k')
            axis.quiver(x_middle,
                      y_middle,
                      scale*dx,
                      scale*dy,
                      scale=5,
                      scale_units='inches')

    # plotting the grid
    plot_bins(my_stats,axis)
    axis.tick_params(labelsize=small_fs)
    
    # title
    if title is not None:
        axis.set_title(title,fontsize=big_fs)
    
    return 0



def plot_partition(partition,axis):
    """
    Plots a 2-dimensional partition as a grid.
    
    INPUT:
        partition: list of rectangles (list of (dim,2) arrays, dim=2 here)
        axis: plt axis (type matplotlib.axes._subplots.AxesSubplot)
        
    """
    # browse the partition and plot each rectangle
    for rect in partition:
        rect_patch = patches.Rectangle((rect[0,0],rect[1,0]),
                                 rect[0,1]-rect[0,0],
                                 rect[1,1]-rect[1,0],
                                 linewidth=lw,
                                 edgecolor='r',
                                 facecolor='none')
        axis.add_patch(rect_patch)
        
    return 0































