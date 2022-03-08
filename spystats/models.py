#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""

Statistical Models
==================

This module combines traversal strategies for visiting and storing the spatial tree-structures.

The spystats range from simple linear to Gaussian Processes.

This file is intented to be big!

Some functions an methods for implementing Gaussian simulation of autocorrelated processes. 

Author :
    Juan Escamilla
    
With Great help of:
     Erick Chacón Montalván    
Date: 
   02/08/2017    
"""

import numpy as np

import scipy as sp
from functools import partial

def corr_exp_ij(distance,phi=1.0):
    """
    This function calculates the correlation function of an exponential model with parameter phi.
    Returns :
        correlation value for distance 'distance'
    notes
    
    """
    return np.exp(-(distance / phi))


def exponentialModel(phi=1.0):
    """
    A functional form of the exponentialModel
    """
    def corr_exp_ij(distance):
        """
        This function calculates the correlation function of an exponential model with parameter phi.
        Returns :
            correlation value for distance 'distance'
        """
        return np.exp(-(distance / phi))
    return corr_exp_ij


 



## This function returns a Distance Matrix given a list of pairs of the form (a,b). It will calculate de distance between (a and b) 

calculateDistanceMatrix = lambda list_of_vectors : np.array(map(lambda a,b : np.linalg.norm(a-b),list_of_vectors))
## note it doesn't have shape of a matrix but doesn't matter.



## Calculate correlations given a valid model 
makeCorrelations = lambda model : lambda list_of_points : np.array(map(model,calculateDistanceMatrix(makeDuples(list_of_points))))


def calculateCovarianceMatrix(points,model,sigma=1.0):
    """
    Returns the covariance matrix calculated from a $Z(x) \sim N(0,|Sigma)$ stationary anisotropic model. 
    """
## Calculate covariances  
    makeCovarianceMatrix = lambda sigma : lambda model: lambda list_of_points : (makeCorrelations(model)(list_of_points) * sigma)#.reshape(100,100)
    
    return makeCovarianceMatrix(sigma)(model)(points)




def exponentialModelFunctional(points,phi,sigma,betas,X):
    """
    Test
    Use the exponential correlation with parameter phi and sigma
    """
    
    n = len(points)
    f = exponentialModel(phi)
    CM = calculateCovarianceMatrix(points, f, sigma=sigma).reshape(n,n)
    weights = betas * X.transpose()
    w = np.array(weights)[0]

    model = sp.stats.multivariate_normal(w.flatten(),CM)
    return model
    
    

def likelihoodFunction(phi,sigma,betas,X,Y_vector,points_location,family='bry'):
    """
    Returns a MVN with parameters "parameters"
    """
    mvn = exponentialModelFunctional(points_location, phi, sigma,betas,X)
    if family =='binary':
        p = mvn.pdf(Y_vector)
        return np.log(p) - np.log( 1 -p)
    else: 
        return mvn.pdf(Y_vector)
    
    
def MlikelihoodFunction(phi_sigma_tuple,X,Y_vector,points_location):
    """
    Returns the minus pdf
    """
    phi,sigma,b1,b2,b3 = phi_sigma_tuple
    betas = np.array([b1,b2,b3])
    p = -1 * likelihoodFunction(phi,sigma,betas,X,Y_vector,points_location)
    return p        
            
            






## Generate the pre-image . For this example will be: time \in R

time = np.linspace(0, 100, 100)
lat = np.linspace(0,100,10)
lon = np.linspace(0,100,10)


### Calculate crosws product of time for distance matrix

makeDuples = lambda list_of_points : [(i,j) for i in list_of_points for j in list_of_points]

points = map(lambda l : np.array(l),makeDuples(lat))


#### Let's generate several correlation functions for different phis

corr_exp_list = map(lambda phi : partial(corr_exp_ij,phi=phi),np.linspace(1,100,50))
corr_exp_list = map(lambda phi : partial(corr_exp_ij,phi=phi),[0.001,20,80])
corr_exp_list = map(lambda phi : partial(corr_exp_ij,phi=phi),[20])

## Different spystats for phi
covarianceMatricesModels = lambda list_of_points : map(lambda model : makeCovarianceMatrix(1.0)(model)(list_of_points),corr_exp_list)

#covarianceMatricesModels = map(lambda model : makeCovarianceMatrix(1.0)(list_of_points)(list_of_points),corr_exp_list)
## Simulation process

zeros = lambda list_of_points : np.zeros(np.sqrt(len(list_of_points))**2)

simulateWithThisPoints = lambda list_of_points : map(lambda Sigma : sp.random.multivariate_normal(zeros(list_of_points),Sigma.reshape(len(list_of_points),len(list_of_points))),covarianceMatricesModels(list_of_points))


## Statistical Models








# from spystats.statistical import *
# t_30_10 = exponentialModelFunctional(points,phi=30,sigma=10)
# phis =np.linspace(20,40,20)
# sigmas =np.linspace(5,15,10)
# phis_sigma = [(phi , sigma) for phi in phis for sigma in sigmas]
# ys = t_30_10.rvs()
# superfunciones = map(lambda (phi,sigma) : likelihoodFunction(phi,sigma,ys,points),phis_sigma)
# A = np.array(superfunciones).reshape(10,20)
# import matplotlib.pyplot as plt
# plt.imshow(A)
# plt.show()
# A = np.array(superfunciones).reshape(20,10)
# plt.imshow(A)
# plt.show()
# superfunciones
# chula = zip(superfunciones,phis_sigma)
# chula
# chula.sort(key=lambda renglon : renglon[0])








