#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
Automatic Batch processing for GP models
========================================
Requires:

    * TensorFlow
    * GPFlow
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
from shapely.geometry import Point
import pandas as pd
#import GPflow as gf
import gpflow as gf
import numpy as np

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# create a file handler
handler = logging.FileHandler('batchprocess.log')
handler.setLevel(logging.INFO)


# create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)


# add the handlers to the logger
logger.addHandler(handler)





def subselectDataFrame(pandasDF, minx,maxx,miny,maxy):
    """
    This function selects a inner region of the DataFrame.
    Receives: A pandas dataframe.
    Returns: A geopandas datafame.
    """
    data = pandasDF
    ## Create the geometric column
    data['geometry'] = data.apply(lambda z: Point(z.LON, z.LAT), axis=1)
    new_data = gpd.GeoDataFrame(data)
    ## Subselect the data
    section = new_data[lambda x:  (x.LON > minx) & (x.LON < maxx) & (x.LAT > miny) & (x.LAT < maxy) ]
    return section
    


def fitMatern12Model(X,Y,optimise=True):
    k = gf.kernels.Matern12(2, lengthscales=1, active_dims = [0,1] )
    model = gf.gpr.GPR(X.as_matrix(),Y.as_matrix().reshape(len(Y),1).astype(float),k)
    model.likelihood.variance = 10
    if optimise:
        model.optimize()
    return model



def buildPredictiveSpace(inputGeoDataFrame,GPFlowModel,num_of_predicted_coordinates=100):
    """
    Creates a grid based on the extent of the inputDataFrame coordinates.
    It uses the information in the geometry column.
    Receives:
        * inputGeoDataFrame : a Geopanda instance with geometry column
        * num_of_predicted_coordinates : Integer, the size of the grid partitioning.
        * GPFlowModel : a GpFlow regressor object (the model that predicts).
    """
    Nn = num_of_predicted_coordinates
    dsc = inputGeoDataFrame
    longitudes  = dsc.apply(lambda c : c.geometry.x, axis=1)
    latitudes = dsc.apply(lambda c : c.geometry.y, axis=1)
    predicted_x = np.linspace(min(longitudes),max(longitudes),Nn)
    predicted_y = np.linspace(min(latitudes),max(latitudes),Nn)
    Xx, Yy = np.meshgrid(predicted_x,predicted_y)
    predicted_coordinates = np.vstack([ Xx.ravel(), Yy.ravel()]).transpose()
    #predicted_coordinate
    means,variances = GPFlowModel.predict_y(predicted_coordinates)

    results = pd.DataFrame([means,variances,longitudes,latitudes])
    return results



def main():
    """
    The main batch processing
    """
    logger.info("Reading Data")
    data = pd.read_csv("/RawDataCSV/idiv_share/plotsClimateData_11092017.csv")
    minx = -90
    maxx = -85 
    miny = 30
    maxy = 35
    logger.info("Subselecting Region")
    section = subselectDataFrame(data, minx, maxx, miny, maxy)
    X = section[['lon','lat']]
    Y = section['SppN']
    logger.info("Fitting GaussianProcess Model")
    model = fitMatern12Model(X, Y, optimise=True)
    logger.info("Predicting Points")  
    space = buildPredictiveSpace(section, model,num_of_predicted_coordinates=300 )
    space.to_csv('test1.csv')
    logger.info("Finished! Results in: tests1.csv")
    
    
    
if __name__ == "__main__":
    main()


## Wishes
#Arguments of geo extent to put during run time
#Arguments for the DataSource
#Arguments for the putput name.
#Atguments maybe for the kernel model.
#For now let's make the test in HEC


## On HEC
#data = pd.read_csv("/home/hpc/28/escamill/csv_data/idiv/plotsClimateData_11092017.csv")





#results.to_csv('test1.csv')


