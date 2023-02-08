# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 10:24:33 2017

@author: m80048406
"""
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(73)
np.set_printoptions(precision=5, linewidth=160, suppress=True)


class DataSet:
    __N_samples = 40
    __x0_mean   = np.array([0,0])
    __x1_mean   = np.array([0,0])
    __x0_cov    = np.array([[1,0], [0,1]])
    __x1_cov    = np.array([[3,0], [0,3]])
    __x_D       = None  
    __y_D       = None
    
    @classmethod
    def get_data(cls):
        if cls.__y_D is None and cls.__x_D is None:
            x0 = np.random.multivariate_normal(cls.__x0_mean, cls.__x0_cov, cls.__N_samples).T
            x1 = np.random.multivariate_normal(cls.__x1_mean, cls.__x1_cov, 2*cls.__N_samples).T
            x1_max = x1.T.dot(-x1).diagonal().argsort()[0:cls.__N_samples]
            x1 = x1[:,x1_max]            
            cls.__x_D = np.concatenate((x0, x1), axis=1)
            y = np.zeros((1,cls.__x_D.shape[1]))
            y[0,cls.__N_samples:]  = 1
            cls.__y_D = y    
        return cls.__y_D, cls.__x_D
    
    @classmethod
    def plot_data(cls):
        y,x = cls.get_data()
        y0_idx = (y[0,:]==0).nonzero()[0]
        y1_idx = (y[0,:]==1).nonzero()[0]
        plt.scatter(x[0,y0_idx], x[1,y0_idx], marker='s', color='b')
        plt.scatter(x[0,y1_idx], x[1,y1_idx], marker='x', color='r')
        plt.axis('equal')    

    @classmethod
    def plot_decision_boundary(cls, predict_f):
        h = .05 
        y,x = cls.get_data()
        x0_min, x0_max = x[0,:].min() - 1, x[0,:].max() + 1
        x1_min, x1_max = x[1,:].min() - 1, x[1,:].max() + 1
        x0_grid, x1_grid = np.meshgrid(np.arange(x0_min, x0_max, h), np.arange(x1_min, x1_max, h))        
        xx = np.vstack([x0_grid.reshape((1,-1)),x1_grid.reshape((1,-1))])            
        a = predict_f(xx).reshape(x0_grid.shape)
        cls.plot_data()
        plt.contour(x0_grid, x1_grid, a, [0.5], colors=('k',))
 
