#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
Tools for analysing spatial data
================================
Requires:
    * Pandas
    * GeoPandas
    * Numpy
    * shapely
    

"""


__author__ = "Juan Escamilla Mólgora"
__copyright__ = "Copyright 2017, JEM"
__license__ = "GPL"
__mantainer__ = "Juan"
__email__ ="molgor@gmail.com"



import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import scipy.spatial as sp
import numpy as np


def toGeoDataFrame(pandas_dataframe,xcoord_name,ycoord_name,srs = 'epsg:4326'):
    """
    Convert Pandas objcet to GeoDataFrame
    Inputs:
        pandas_dataframe : the pandas object to spatialise
        xcoord_name : (String) the column name of the x coordinate.
        ycoord_name : (String) the column name of the y coordinate. 
        srs : (String) the source referencing system in EPSG code.
                e.g. epsg:4326 .
    """
    data = pandas_dataframe
    data['geometry'] = data.apply(lambda z : Point(z[xcoord_name], z[ycoord_name]), axis=1)
    #data['geometry'] = data.apply(lambda z : Point(z.LON, z.LAT), axis=1)

    new_data = gpd.GeoDataFrame(data)
    new_data.crs = {'init':'epsg:4326'}
    return new_data




def _getDistanceMatrix(geopandas_dataset):
    """
    Returns the "self/auto" distance matrix of a given list of vector data. 
    By default it uses the Minkowski distance of order 2.
    Parameters :
        p : the Minkowski distance exponent (order)
    """
    data = geopandas_dataset
    coords = zip(data.centroid.x,data.centroid.y)
    dM = sp.distance_matrix(coords,coords,p=2.0)
    return dM
    
    
    
def _getDistResponseVariable(geopandas_dataset,response_variable_name):
    """ 
    Returns the "self/auto" distance matrix of a response variable Y 
    By default it uses the Minkowski distance of order 2.
    meaning:
    $$ v_{i,j} = \sum_{i=1}^{n} (y_i - y_j)^2 $$
    Parameters :
        geopandas_dataset : (geopandas) the geopandas dataframe.
        response_variable_name : (string) name of the variable for calculating the distance.
        p : the Minkowski distance exponent (order)
    """
    data = geopandas_dataset
    y = data[response_variable_name]
    yy = y.values.reshape(-1,1)
    dY = sp.distance_matrix(yy,yy,p=2.0)
    return dY
 
def calculateEmpiricalVariogram(distances,response_variable,n_bins=50):
    """
    Returns the empirical variogram given by the formula
    $$ v_{lag_i} = \frac{\sum_{i=1}^{N(lag_i)} (y_i - y_j)^2}{2} $$    
    Parameters:
    n_bins : (Integer) number of bins (lag distances)  
    """
    mdist = min(distances)
    Mdist = max(distances)
    partitions = np.linspace(mdist,Mdist,n_bins)
    lags = partitions[:n_bins - 1]
    y = response_variable
    d = pd.DataFrame({'dist': distances,'y':y})
      
    # The actual emp. var function     
    empvar =  map(lambda (i,x) : 0.5 * (d[ ( d.dist < partitions[i+1]) & (d.dist>partitions[i])].y.mean()),enumerate(lags))
    #self.empirical = empvar
    results = pd.DataFrame({'lags':lags,'variogram':empvar})
    
    return results  
 

def montecarloEnvelope(distances,response_variable,num_iterations=99):
    """
    Generate Monte Carlo envelope by shuffling the response variable and keeping the distances the same.
    After GeoR. ("Model-Based Geostatistics; Diggle and Ribeiro, 2007")
    Parameters :
        distances (List) linearised format of a distance matrix
        response_variable (list) linearised format of a response distance matrix.
    """
    simulation_variograms = []
    d = calculateEmpiricalVariogram(distances,response_variable)
    for i in range(num_iterations):
        #np.random.shuffle(response_variable)
        d = calculateEmpiricalVariogram(distances,response_variable)
        simulation_variograms.append(d.variogram)
        np.random.shuffle(response_variable)

    simulation_variograms.append(d.lags)
    #sims = pd.DataFrame(simulation_variograms).transpose()
    sims = pd.DataFrame(simulation_variograms)

    #sims.set_index('lags')
    
    low_q = sims.quantile(0.025)
    high_q = sims.quantile(0.975)
    envelope = pd.DataFrame({'envlow':low_q,'envhigh':high_q,'lags':d.lags})
    return envelope    

class Variogram(object):
    """
    A class that defines Empirical Variogram objects.
    """
    def __init__(self,geopandas_dataset,response_variable_name):
        """
        Constructor
            Parameters :
                geopandas_dataset : (geopandas) the geopandas dataframe.
                response_variable_name : (string) name of the variable for calculating the distance.
                p : the Minkowski distance exponent (order)
        """
        self.distance_coordinates = _getDistanceMatrix(geopandas_dataset)
        self.distance_responses = _getDistResponseVariable(geopandas_dataset,response_variable_name)
        self.empirical = []
        self.lags = []
        self.envelope = []
  
    
    
        
    def calculate_empirical(self,n_bins=50):
        """
        Returns the empirical variogram given by the formula:
        $$ v_{lag_i} = \frac{\sum_{i=1}^{N(lag_i)} (y_i - y_j)^2}{2} $$
        
        Parameters:
            n_bins : (Integer) number of bins (lag distances) 
        
        This only assigns the data to the corresponding attributes.
        
        """
        
        distances = self.distance_coordinates.flatten()
        y = self.distance_responses.flatten()
        results = calculateEmpiricalVariogram(distances,y,n_bins=n_bins)
        
        self.lags = results.lags
        self.empirical = results.variogram
        return self.empirical
        
    def calculateEnvelope(self,num_iterations=99):
        """
        Calculates the Montecarlo variogram envelope.
        """
        distances = self.distance_coordinates.flatten()
        responses = self.distance_responses.flatten()
        envelopedf = montecarloEnvelope(distances,responses,num_iterations=num_iterations)
        envelopedf = pd.concat([envelopedf,self.empirical],axis=1)
        self.envelope = envelopedf
        return envelopedf