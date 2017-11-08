#!/usr/bin/env python
#-*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals


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
import matplotlib.pyplot as plt
import logging
#from external_plugins.spystats import tools as tl



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)







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
 

def montecarloEnvelope(distances,response_variable,num_iterations=99,n_bins=50):
    """
    Generate Monte Carlo envelope by shuffling the response variable and keeping the distances the same.
    After GeoR. ("Model-Based Geostatistics; Diggle and Ribeiro, 2007")
    Parameters :
        distances (List) linearised format of a distance matrix
        response_variable (list) linearised format of a response distance matrix.
    """
    simulation_variograms = []
    d = calculateEmpiricalVariogram(distances,response_variable,n_bins=n_bins)
    for i in range(num_iterations):
        #np.random.shuffle(response_variable)
        d = calculateEmpiricalVariogram(distances,response_variable,n_bins=n_bins)
        simulation_variograms.append(d.variogram)
        np.random.shuffle(response_variable)

    #simulation_variograms.append(d.lags)
    #sims = pd.DataFrame(simulation_variograms).transpose()
    sims = pd.DataFrame(simulation_variograms)

    #sims.set_index('lags')
    
    low_q = sims.quantile(0.025)
    high_q = sims.quantile(0.975)
    envelope = pd.DataFrame({'envlow':low_q,'envhigh':high_q,'lags':d.lags})
    return (envelope,sims)
    #return envelope    

class Variogram(object):
    """
    A class that defines Empirical Variogram objects.
    """
    def __init__(self,geopandas_dataset,response_variable_name,using_distance_threshold=False):
        """
        Constructor
            Parameters :
                geopandas_dataset : (geopandas) the geopandas dataframe.
                response_variable_name : (string) name of the variable for calculating the distance.
                p : the Minkowski distance exponent (order)
        """
        self.distance_coordinates = _getDistanceMatrix(geopandas_dataset)
        self.distance_responses = _getDistResponseVariable(geopandas_dataset,response_variable_name)
        self.empirical = pd.Series()
        self.lags = []
        self.envelope = pd.DataFrame()
  
    
    
        
    def calculateEmpirical(self,n_bins=50):
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
        
    def calculateEnvelope(self,num_iterations=99,n_bins=50):
        """
        Calculates the Montecarlo variogram envelope.
        """
        logger.info("Calculating envelope via MonteCarlo Simulations. \n Using %s iterations"%num_iterations)
        distances = self.distance_coordinates.flatten()
        responses = self.distance_responses.flatten()
        envelopedf,sims = montecarloEnvelope(distances,responses,num_iterations=num_iterations,n_bins=n_bins)
        envelopedf = pd.concat([envelopedf,self.empirical],axis=1)
        self.envelope = envelopedf
        return envelopedf
    
    
    def plot(self,with_envelope=True,percentage_trunked=10,refresh=True,n_bins=50,plot_filename=False,**kwargs):
        """
        Plot the empirical semivariogram with optional confidence interval using MonteCarlo permutations at 0.025 and 0.975 quantiles.
        Returns a matplotlib object.
        Parameters : 
            with_envelope : (Boolean) if true it will calculate and plot the 0.025 and 0.975 quantiles of a montecarlo permutations of $Y$ using fixed locations.
            percentage_trunked = (float) Percentage of data removed in the plot. This is to ease the visualisation by cutting the last values
                    
        Extra parameters in the kwargs
            * num_iterations : (Integer) see CalculateEnvelope
            * n_bins : (Integer) see calculate_variogram 
        
        """
        
        
        #v = env.iloc[1:30,:]
        #points = plt.scatter(vg.lags,vg.empirical)
        if (self.empirical.empty or refresh == True):
            logger.info("Calculating empirical variogram")
            self.calculateEmpirical(n_bins=n_bins)
        
        nrows = self.empirical.shape[0]
        indx = int(np.ceil(float(percentage_trunked)/100 * nrows))
        
        lags = self.lags.iloc[: (nrows - indx)]
        empirical = self.empirical.iloc[:(nrows - indx)]
        
                           
        if with_envelope:
            if ( self.envelope.empty or refresh == True) :
                logger.info("No envelope object found. Calculating...")
                num_iter = kwargs.get('num_iterations')
                if isinstance(num_iter, int):
                    self.calculateEnvelope(num_iterations=num_iter,n_bins=n_bins)
                else:
                    self.calculateEnvelope()
                #except:
                #    self.calculateEnvelope()
            else:
                logger.info("Using previously stored envelope. Use refresh option to recalculate.")   
            
            envelope = self.envelope.iloc[:(nrows - indx)]
            
            ## ********* PLOT    
            #plt.plot(lags,empirical,'o--',lw=2.0)
            ### ***** PLOT
            plt.plot(lags,envelope.envhigh,'k--')
            plt.plot(lags,envelope.envlow,'k--')
            plt.fill_between(lags,envelope.envlow,envelope.envhigh,alpha=0.5)
            plt.legend(labels=['97.5%','emp. varig','2.5%'])
        
        
        ## ********* PLOT    
        plt.plot(lags,empirical,'o--',lw=2.0)         
        ## ****** PLOT
        plt.legend(loc='best')
        plt.xlabel("Distance in meters")
        plt.ylabel("Semivariance")
        #plt.legend(labels=['97.5%','emp. varig','2.5%'])
        #ax = 
        #points2 = plt.lines(vg.lags,vg.empirical,c='red')
        #plt.show()
        logger.debug("Check which object to return. maybe a figure")
        if plot_filename :
            plt.savefig(plot_filename)
        
        return None 
        
        
        
if __name__ == "__main__":
    __package__ = "spystats"
    




        