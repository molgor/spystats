#!/usr/bin/env python
#-*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

"""
Test script for calculating empirical semivariogram and its envelope.
    

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

## Use the ggplot style
plt.style.use('ggplot')

from external_plugins.spystats import tools
# My mac
logging.info("Reading data")
#data = pd.read_csv("/RawDataCSV/plotsClimateData_11092017.csv")
# My Linux desktop
data = pd.read_csv("/RawDataCSV/idiv_share/plotsClimateData_11092017.csv")
new_data = tools.toGeoDataFrame(pandas_dataframe=data,xcoord_name='LON',ycoord_name='LAT')
## Reproject to alberts
new_data =  new_data.to_crs("+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=37.5 +lon_0=-96 +x_0=0 +y_0=0 +ellps=GRS80 +datum=NAD83 +units=m +no_defs ")
new_data['logBiomass'] = new_data.apply(lambda x : np.log(x.plotBiomass),axis=1)
new_data['newLon'] = new_data.apply(lambda c : c.geometry.x, axis=1)
new_data['newLat'] = new_data.apply(lambda c : c.geometry.y, axis=1)
#new_data.plot(column='SppN')
new_data['logBiomass'] = np.log(new_data.plotBiomass)
new_data['logSppN'] = np.log(new_data.SppN)


#new_data.logBiomass.plot.hist()

#linear model
model = smf.ols(formula='logBiomass ~ SppN',data=new_data)
results = model.fit()
param_model = results.params
results.summary()

new_data['residuals1'] = results.resid


model2 = smf.ols(formula='logBiomass ~ logSppN',data=new_data)
results2 = model.fit()
param_model = results2.params
results2.summary()
new_data['residuals2'] = results2.resid



## Select a section
#section = new_data[lambda x:  (x.LON > -90) & (x.LON < -85) & (x.LAT > 30) & (x.LAT < 35) ]
#vg = tools.Variogram(new_data,'logBiomass')
#vg.calculate_empirical(n_bins=10)



def subselectDataFrameByCoordinates(dataframe,namecolumnx,namecolumny,minx,maxx,miny,maxy):
    """
    Returns a subselection by coordinates using the dataframe/
    """
    minx = float(minx)
    maxx = float(maxx)
    miny = float(miny)
    maxy = float(maxy)
    section = dataframe[lambda x:  (x[namecolumnx] > minx) & (x[namecolumnx] < maxx) & (x[namecolumny] > miny) & (x[namecolumny] < maxy) ]
    return section




def getExtent(geodataframe):
    """
    REturn the tuple of the spatial extent. Based on geopandas geometry attribute.
    """
    minx = min(geodataframe.geometry.x)
    maxx = max(geodataframe.geometry.x)

    miny = min(geodataframe.geometry.y)
    maxy = max(geodataframe.geometry.y)
    

    return (minx,maxx,miny,maxy)
  

def getExtentFromPoint(x,y,step_sizex,step_sizey):
    """
    Returns a tuple (4) specifying the minx,maxx miny, maxy based on a given point and a step size.
    The x,y point is located in the bottom left corner.
    """  
    minx = x
    miny = y
    maxx = x + step_sizex
    maxy = y + step_sizey
    return (minx,maxx,miny,maxy)
    
    
  
    
section = subselectDataFrameByCoordinates(new_data, 'lon', 'lat', -90 , -85,30, 35)







