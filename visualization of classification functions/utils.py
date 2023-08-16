#------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------function to create synthetic dataset----------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------

from sklearn.datasets import make_blobs, make_circles, make_moons, make_classification

n_samples=2000

def create_dataset(method ,n_samples=n_samples, random_state=24, **kwargs ):
    '''
    INPUT: -method: one of following stings to call ["make_blobs2", "make_blobs2", "make_circles", "make_moons", "make_classification"] to generate datasets
           -n_samples: initialized to 2000
           -random_state initialized to 24
           -**kwargs
   OUTPUT: -returns a tuple of (X, y) fo 2 numpy.arrays of shapes (n_samples, 2) and (n_samples)    
    '''
    
    
    if method == 'make_blobs3':
        return make_blobs(n_samples=n_samples, random_state= random_state, centers=3, cluster_std = 2)
    if method == 'make_blobs2':
        return make_blobs(n_samples=n_samples, random_state= random_state, centers=2, cluster_std = 2)
    elif method =='make_circles':
        return make_circles(n_samples=n_samples, random_state=random_state, noise=0.2)
    elif method =='make_classification':
         return make_classification(n_samples=n_samples, random_state =random_state, n_features = 2, n_redundant=0 , n_classes=2)
    elif method =='make_moons':
        return make_moons(n_samples = n_samples, random_state = random_state, noise = 0.2 )
    
    
#----------------------------------------------------------------------------------------------------------------------------    
#-------------------------------------------------function to plot raw dataset-----------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------


import matplotlib as mpl
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def plot_data (X=None, y=None, alpha = 1, training_testing = False, training_data = None, testing_data = None):
    '''
    function to plot data points either from the training set or the testing set or the full dataset.
    
    '''
    
    if training_testing == True:
        
        fig, ax = plt.subplots(1, 2, figsize=(12,6))
        #fig.figsize = (12,14)
        for label in range(np.unique(training_data[1]).shape[0] ):
            mask = (training_data[1] == label)
            ax[0].scatter(training_data[0][mask, 0], training_data[0][mask, 1] , label="train_y={}".format(label), alpha = alpha)
            ax[0].legend()
        for label in range(np.unique(testing_data[1]).shape[0] ):
            mask = (testing_data[1] == label)
            ax[1].scatter(testing_data[0][mask, 0], testing_data[0][mask, 1] , label="test_y={}".format(label), alpha = alpha*.2)
            ax[1].legend()
            #ax[0].title("Training dataset")
            #ax[1].title("Testing dataset")
    else:  
        fig, ax = plt.subplots()
        fig.figsize = (9,6)
        for label in range(np.unique(y).shape[0] ):
            mask = (y == label)
            ax.scatter(X[mask, 0], X[mask, 1] , label="y={}".format(label), alpha = alpha)
            plt.legend()
            plt.title("Full dataset")

        
#-------------------------------------------------------------------------------------------------------------------------------        
#------------------------------------------functions to plot classifications regions--------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------
        
from matplotlib.colors import ListedColormap
from sklearn.inspection import DecisionBoundaryDisplay


def classification_plot(clf_dict, X ,y, acc_scores_dict):
    '''
    function to plot decision areas for each category, as defined by the classifier
    
    '''
    
    fig, ax = plt.subplots(2, 3, figsize = (20,8))
    num_labels = np.unique(y).shape[0]
    i=0
    for key, clf in clf_dict.items():
        
        if i <=2:
            DecisionBoundaryDisplay.from_estimator(clf, X, ax=ax[0, i],  alpha = 0.3 ,plot_method="contourf", 
                                       cmap = ListedColormap(['#1f77b4','#ff7f0e','#2ca02c','#d62728',\
                                                              '#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22', '#17becf'][:num_labels]))
            DecisionBoundaryDisplay.from_estimator(clf, X, ax=ax[0,i],  alpha = 0.3 ,plot_method="contour", 
                                       cmap = ListedColormap(['#1f77b4','#ff7f0e','#2ca02c','#d62728',\
                                                              '#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22', '#17becf'][:num_labels]))


            for label in range(num_labels ):
                mask = (y == label)
                ax[0,i].scatter(X[mask, 0], X[mask, 1] , label="y={}".format(label),)
                #plt.legend()
            ax[0,i].set_title(key+'\n'+'score'+': '+str(acc_scores_dict[key]))
            i=i+1
        
            
        else :
            DecisionBoundaryDisplay.from_estimator(clf, X, ax=ax[1, i-3],  alpha = 0.3 ,plot_method="contourf", 
                                       cmap = ListedColormap(['#1f77b4','#ff7f0e','#2ca02c','#d62728',\
                                                              '#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22', '#17becf'][:num_labels]))
            DecisionBoundaryDisplay.from_estimator(clf, X, ax=ax[1,i-3],  alpha = 0.3 ,plot_method="contour", 
                                       cmap = ListedColormap(['#1f77b4','#ff7f0e','#2ca02c','#d62728',\
                                                              '#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22', '#17becf'][:num_labels]))


            for label in range(num_labels ):
                mask = (y == label)
                ax[1, 3-i].scatter(X[mask, 0], X[mask, 1] , label="y={}".format(label),)
                #plt.legend()
            ax[1,3-i].set_title(key+'\n'+'score'+': '+str(acc_scores_dict[key]))        
            i = i+1
            