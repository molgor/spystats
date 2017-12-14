#!/usr/bin/env python
#-*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

"""
Script for fitting a GLS based on a theoretical variogram fitting.
"""


__author__ = "Juan Escamilla Mólgora"
__copyright__ = "Copyright 2017, JEM"
__license__ = "GPL"
__mantainer__ = "Juan"
__email__ ="molgor@gmail.com"


import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import statsmodels.formula.api as smf
import sys
sys.path.append('..')
sys.path.append('/home/hpc/28/escamill/spystats')
import tools
import statsmodels.regression.linear_model as lm
import statsmodels.api as sm
import logging

## Use the ggplot style
plt.style.use('ggplot')

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# create a file handler
handler = logging.FileHandler('batchprocess_log_log.log')
handler.setLevel(logging.INFO)


# create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)


# add the handlers to the logger
logger.addHandler(handler)


### PATH information
#plotdata_path = "/RawDataCSV/idiv_share/plotsClimateData_11092017.csv"
#empirical_data_path = "../HEC_runs/results/logbiomas_logsppn_res.csv"







########## Experimental
def main(plotdata_path,empirical_data_path,minx,maxx,miny,maxy):
    """
    The main batch processing
    """
    
    data = pd.read_csv(csv_path)
    minx = minx
    maxx = maxx
    miny = miny
    maxy = maxy
    
    logging.info("Reading data")

    data = pd.read_csv(plotdata_path)
    new_data = tools.toGeoDataFrame(pandas_dataframe=data,xcoord_name='LON',ycoord_name='LAT')
    ## Reproject to alberts
    logger.info("Reprojecting to Alberts equal area")
    new_data =  new_data.to_crs("+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=37.5 +lon_0=-96 +x_0=0 +y_0=0 +ellps=GRS80 +datum=NAD83 +units=m +no_defs ")
    new_data['logBiomass'] = new_data.apply(lambda x : np.log(x.plotBiomass),axis=1)
    new_data['newLon'] = new_data.apply(lambda c : c.geometry.x, axis=1)
    new_data['newLat'] = new_data.apply(lambda c : c.geometry.y, axis=1)
    new_data['logBiomass'] = np.log(new_data.plotBiomass)
    new_data['logSppN'] = np.log(new_data.SppN)
    
    #linear model
    logger.info("Fitting OLS linear model: logBiomass ~ logSppN ")
    model = smf.ols(formula='logBiomass ~ logSppN',data=new_data)
    results = model.fit()
    param_model = results.params
    results.summary()
    new_data['residuals'] = results.resid
    
    logger.info("Removing possible duplicates. \n This avoids problems of Non Positive semidefinite")
    new_data = new_data.drop_duplicates(subset=['newLon','newLat'])
    
    ## Read the empirical variogram
    logger.info("Reading the empirical Variogram file")
    thrs_dist = 100000
    
    ## Change here with appropriate path for file
    empirical_semivariance_log_log = empirical_data_path
    #### here put the hec calculated 
    logger.info("Instantiating a Variogram object with the values calculated before")
    emp_var_log_log = pd.read_csv(empirical_semivariance_log_log)
    gvg = tools.Variogram(new_data,'logBiomass',using_distance_threshold=thrs_dist)
    gvg.envelope = emp_var_log_log
    gvg.empirical = emp_var_log_log.variogram
    gvg.lags = emp_var_log_log.lags
    logger.info("Dropping possible Nans")
    emp_var_log_log = emp_var_log_log.dropna()
    vdata = gvg.envelope.dropna()
    
    logger.info("Instantiating Whittle Model...")
    whittle_model = tools.WhittleVariogram(sill=0.345,range_a=100000,nugget=0.33,alpha=1.0)
    logger.info("fitting Whittle Model with the empirical variogram")
    tt = gvg.fitVariogramModel(whittle_model)
    logger.info("Whittle Model fitted")
    

    
    
    
    
    
    logger.info("Subselecting Region")
    
    ## new code
    section = tools._subselectDataFrameByCoordinates(new_data,'LON','LAT',minx,maxx,miny,maxy)
    secvg = tools.Variogram(section,'logBiomass',model=whittle_model)
    logger.info("Calculating Distance Matrix")
    MMdist = secvg.distance_coordinates.flatten()
    logger.info("Calculating Correlation based on theoretical model")
    CovMat = secvg.model.corr_f(MMdist).reshape(len(section),len(section))
    
    logger.info("Fitting linear model using GLS")
    model1 = lm.GLS.from_formula(formula='logBiomass ~ logSppN',data=section,sigma=CovMat)
    results = model1.fit()
    resum = results.summary()
    logger.info("Writing to file")
    f = open("test_gls.csv",'w')
    f.write(resum.as_text())
    f.close()    
    
    logger.info("Finished! Results in: tests1.csv")
    
    
    
if __name__ == "__main__":
    plotdata_path = sys.argv[1]
    empirical_data_path = sys.argv[2]
    minx = float(sys.argv[3])
    maxx = float(sys.argv[4])
    miny = float(sys.argv[5])
    maxy = float(sys.argv[6])
    predicted_size = float(sys.argv[7])
    main(plotdata_path,empirical_data_path,minx,maxx,miny,maxy,predicted_size)




## Sectionate by coordinates
##minx = -85
##maxx = -80
##miny = 30
##maxy = 35
### PATH information
#plotdata_path = "/RawDataCSV/idiv_share/plotsClimateData_11092017.csv"
#empirical_data_path = "./HEC_runs/results/logbiomas_logsppn_res.csv"
## For HEC path in /home/hpc/28/escamill/csv_data/idiv/plotsClimateData_11092017.csv

## For running ALL DATA SET
## python fit_fia_logbiomass_logspp_GLS /home/hpc/28/escamill/csv_data/idiv/plotsClimateData_11092017.csv ./HEC_runs/results/logbiomas_logsppn_res.csv -90 -80 30 40  

## Test small region
## python fit_fia_logbiomass_logspp_GLS /home/hpc/28/escamill/csv_data/idiv/plotsClimateData_11092017.csv ./HEC_runs/results/logbiomas_logsppn_res.csv -90 -80 30 40  


## Checar la importacion de la libreria spystats



#python fit_fia_sppn.py /RawDataCSV/idiv_share/plotsClimateData_11092017.csv -85 -80 30 35  
## In hec
#python fit_fia_sppn.py /home/hpc/28/escamill/csv_data/idiv/plotsClimateData_11092017.csv -90 -80 30 40 300 

