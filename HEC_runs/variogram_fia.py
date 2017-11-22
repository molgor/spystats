#!/usr/bin/env python
#-*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals


"""
Module for calculating a variogram. 
This is not interactive
The output will be written in the base folder in CSV format.
"""


__author__ = "Juan Escamilla Mólgora"
__copyright__ = "Copyright 2017, JEM"
__license__ = "GPL"
__mantainer__ = "Juan"
__email__ ="molgor@gmail.com"


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import logging
import sys
sys.path.append('..')
sys.path.append('/home/hpc/28/escamill/spystats')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from tools import toGeoDataFrame, Variogram


def main(csv_path,minx,maxx,miny,maxy,predicted_size=300,distance_threshold=500000):
    """
    The main batch processing
    """
    ### Init configuration
    
    ## Use the ggplot style
    
    
    
    plt.style.use('ggplot')
    
    
    
    logger.info("Reading data")
    
    
    # My mac
    #data = pd.read_csv("/RawDataCSV/plotsClimateData_11092017.csv")
    # My Linux desktop
    
    data = pd.read_csv(csv_path)
    #data = pd.read_csv("/RawDataCSV/idiv_share/plotsClimateData_11092017.csv")
    #new_data = tools.toGeoDataFrame(pandas_dataframe=data,xcoord_name='LON',ycoord_name='LAT')

    new_data = toGeoDataFrame(pandas_dataframe=data,xcoord_name='LON',ycoord_name='LAT')

    logger.info("Performing Reprojection to Alberts")
    ## Reproject to alberts
    new_data =  new_data.to_crs("+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=37.5 +lon_0=-96 +x_0=0 +y_0=0 +ellps=GRS80 +datum=NAD83 +units=m +no_defs ")
    new_data['logBiomass'] = new_data.apply(lambda x : np.log(x.plotBiomass),axis=1)
    new_data['newLon'] = new_data.apply(lambda c : c.geometry.x, axis=1)
    new_data['newLat'] = new_data.apply(lambda c : c.geometry.y, axis=1)
    #new_data.plot(column='SppN')
    new_data['logBiomass'] = np.log(new_data.plotBiomass)
    new_data['logSppN'] = np.log(new_data.SppN)
    #new_data.logBiomass.plot.hist()
    
    logger.info("Fitting Linear Model")
    #linear model
    model = smf.ols(formula='logBiomass ~ logSppN',data=new_data)
    results = model.fit()
    param_model = results.params
    results.summary()
    
    new_data['residuals1'] = results.resid
    
    
    logger.info("Cliping data")
    ## Select a section
    
    
    section = new_data[lambda x:  (x.LON > minx) & (x.LON < maxx) & (x.LAT > miny) & (x.LAT < maxy) ]
    #vg = tools.Variogram(new_data,'logBiomass')
    #vg.calculate_empirical(n_bins=10)
    
    logger.info("Calculating Empirical Variogram")
    #vg = tools.Variogram(section,'residuals1')
    vg = Variogram(section,'residuals1',using_distance_threshold=distance_threshold)
    
    #vg.calculate_empirical(n_bins=50)
    vgplot = vg.plot(num_iterations=40,n_bins=50,plot_filename='test1.png')
    logger.info("Copying to output file")
    vg.envelope.to_csv("data_envelope.csv")
    logger.info("Finished! Results in: tests1.csv")
    

    
    
    
if __name__ == "__main__":
    __package__ = "spystats"
    csv_path = sys.argv[1]
    minx = float(sys.argv[2])
    maxx = float(sys.argv[3])
    miny = float(sys.argv[4])
    maxy = float(sys.argv[5])
    dist_thres = float(sys.argv[6])
    
    main(csv_path,minx,maxx,miny,maxy,distance_threshold=dist_thres)


