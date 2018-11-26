#!/usr/bin/env python
#-*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

"""
Script for fitting a GLS based on a theoretical variogram fitting.

There are custum things hard wired. This should be refactored
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

def systSelection(dataframe,k):
    n = len(dataframe)
    idxs = range(0,n,k)
    systematic_sample = dataframe.iloc[idxs]
    return systematic_sample



def prepareDataFrame(empirical_data_path):
    """
    Reads the data and stores it in a geodataframe.
    """
    logging.info("Reading data")
    data = pd.read_csv(empirical_data_path)
    new_data = tools.toGeoDataFrame(pandas_dataframe=data,xcoord_name='LON',ycoord_name='LAT')
    ## Reproject to alberts
    logger.info("Reprojecting to Alberts equal area")
    new_data =  new_data.to_crs("+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=37.5 +lon_0=-96 +x_0=0 +y_0=0 +ellps=GRS80 +datum=NAD83 +units=m +no_defs ")
    new_data['logBiomass'] = new_data.apply(lambda x : np.log(x.plotBiomass),axis=1)
    new_data['newLon'] = new_data.apply(lambda c : c.geometry.x, axis=1)
    new_data['newLat'] = new_data.apply(lambda c : c.geometry.y, axis=1)
    new_data['logBiomass'] = np.log(new_data.plotBiomass)
    new_data['logSppN'] = np.log(new_data.SppN)
    logger.info("Removing possible duplicates. \n This avoids problems of Non Positive semidefinite")
    new_data = new_data.drop_duplicates(subset=['newLon','newLat'])
    return new_data

def fitLinearLogLogModel(geodataframe):
    """
    This is a stupid hardcoded function.
    """
    #linear model
    logger.info("Fitting OLS linear model: logBiomass ~ logSppN ")
    model = smf.ols(formula='logBiomass ~ logSppN',data=geodataframe)
    results = model.fit()
    param_model = results.params
    #summary = results.summary()
    return (model,results)


def loadVariogramFromData(plotdata_path,geodataframe):
    """
    Reads and instantiates a Variogram object using data stored in plotdata_path.
    """    

    ## Read the empirical variogram
    logger.info("Reading the empirical Variogram file")
    thrs_dist = 1000000
    
    ## Change here with appropriate path for file
    empirical_semivariance_log_log = plotdata_path
    #### here put the hec calculated 
    logger.info("Instantiating a Variogram object with the values calculated before")
    emp_var_log_log = pd.read_csv(empirical_semivariance_log_log)
    gvg = tools.Variogram(geodataframe,'logBiomass',using_distance_threshold=thrs_dist)
    gvg.envelope = emp_var_log_log
    gvg.empirical = emp_var_log_log.variogram
    gvg.lags = emp_var_log_log.lags
    logger.info("Dropping possible Nans")
    emp_var_log_log = emp_var_log_log.dropna()
    vdata = gvg.envelope.dropna()

    logger.info("Instantiating Model...")
    #matern_model = tools.MaternVariogram(sill=0.34,range_a=100000,nugget=0.33,kappa=0.5)
    whittle_model = tools.WhittleVariogram(sill=0.34,range_a=100000,nugget=0.0,alpha=3)
    logger.info("fitting %s Model with the empirical variogram"%whittle_model.name)
    gvg.model = whittle_model
    tt = gvg.fitVariogramModel(whittle_model)
    logger.info("Model fitted")  
    return (gvg,tt)


def buildSpatialStructure(geodataframe,theoretical_model):
    """
    wrapper function for calculating spatial covariance matrix
    """
    secvg = tools.Variogram(geodataframe,'logBiomass',model=theoretical_model)
    logger.info("Calculating Distance Matrix")
    CovMat = secvg.calculateCovarianceMatrix()
    return CovMat

def calculateGLS(geodataframe,CovMat):
    logger.info("Fitting linear model using GLS")
    try:
        model1 = lm.GLS.from_formula(formula='logBiomass ~ logSppN',data=geodataframe,sigma=CovMat)
    except np.linalg.LinAlgError as err:
        logger.warn("Non positive definite matrix")
        results = np.nan 
        resum = "No convergence. Error (%s)"%str(err)
        return (results,resum)
    results = model1.fit()
    resum = results.summary()
    return (results,resum)


def bundleToGLS(geodataframe,theoretical_model):
    CovMat = buildSpatialStructure(geodataframe, theoretical_model)
    results,resum = calculateGLS(geodataframe, CovMat)
    #return results
    return results.nobs, results.rsquared, results.params, results.pvalues, results.conf_int()


def analysisForNestedSystematicSampling(geodataframe_whole,variogram):
    
    samples = map(lambda i : systSelection(geodataframe_whole,i), range(20,2,-1))
    logger.info("Initializing systematic sampling")
    lparams = []
    lpvals = []
    lrsq = []
    lconf_int = []
    ln_obs = []
    for sample in samples:
        n_obs, rsq,params,pvals,conf_int = bundleToGLS(sample,variogram.model)
        logger.info("RESULTS::: n_obs: %s, r-squared: %s, {%s,%s,%s}"%(n_obs,rsq,params.to_json(),pvals.to_json(),conf_int.to_json()))
        ln_obs.append(n_obs)
        lrsq.append(rsq)
        lparams.append(params)
        lpvals.append(pvals)
        lconf_int.append(conf_int)
    return ln_obs,lrsq,lparams,lpvals,lconf_int





def initAnalysis(empirical_data_path,plotdata_path,minx,maxx,miny,maxy):
    """
    Initialises the data.
    Prepares it,
    Append residuals column using OLS.
    """
    ## make subsection
    new_data = prepareDataFrame(empirical_data_path)
    model, results = fitLinearLogLogModel(new_data)
    new_data['residuals'] = results.resid
    logger.info("Subselecting Region")
    section = tools._subselectDataFrameByCoordinates(new_data,'LON','LAT',minx,maxx,miny,maxy)
    return section
    
def fitGLSRobust(geodataframe,variogram_object,num_iterations=20,distance_threshold=1000000):
    """
    Fits a GLS model iterating through the residuals of the previous GLS.
    After estimating the parameters the method uses the new residuals as input.
    Recalculates the empirical variogram, refits the theoretical variogram, recalculates de Covariance Matrix.
    This is done `num_iterations` times.
    
    note: the geodataframe needs to have a column called `residuals` 
    """
    lrsq = []
    lparams = []
    lpvals = []
    lconf_int = []
    iterations = True
    variogram = variogram_object.model
    for i in range(num_iterations):    
        logger.info("Building Spatial Covariance Matrix")
        CovMat = buildSpatialStructure(geodataframe,variogram_object.model)
        logger.info("Calculating GLS estimators")
        results,resum = calculateGLS(geodataframe,CovMat)
        

        try:
            logger.warn("results %s"%results)
            #
            n_obs = results.nobs
            rsq = results.rsquared
            params = results.params
            pvals = results.pvalues 
            conf_int = results.conf_int()
        except:
            n_obs = np.nan
            rsq = np.nan
            params = np.nan
            pvals = np.nan
            conf_int = np.nan 
            resultspd = pd.DataFrame( {'rsq' : lrsq, 'params': lparams, 'pvals' : lpvals, 'conf_int' : lconf_int})
            logger.warn("Nothing to fit. No aparent spatial autocorrelation")
            return (resum,variogram_object,resultspd,results)
        
        lrsq.append(rsq)
        lparams.append(params)
        lpvals.append(pvals)
        lconf_int.append(conf_int)
        logger.info("RESULTS::: n_obs: %s, r-squared: %s, {%s,%s,%s}"%(n_obs,rsq,params.to_json(),pvals.to_json(),conf_int.to_json()))
        geodataframe.residuals = results.resid
        envelope = variogram_object.envelope
        variogram_object = tools.Variogram(geodataframe,'residuals',using_distance_threshold=distance_threshold,model=variogram_object.model)
        variogram_object.envelope = envelope
        #variogram_object.empirical = empirical
        logger.info("Recalculating variogram")
        variogram_object.calculateEmpirical()
        logger.info("Refitting Theoretical Variogram")
        tt = variogram_object.fitVariogramModel(variogram_object.model)
        logger.info("Variogram parameters: range %s, sill %s, nugget %s"%(tt.range_a,tt.sill,tt.nugget))

    resultspd = pd.DataFrame( {'rsq' : lrsq, 'params': lparams, 'pvals' : lpvals, 'conf_int' : lconf_int})
    return (resum,variogram_object,resultspd,results)
    
########## Experimental
def main(empirical_data_path,plotdata_path,minx,maxx,miny,maxy):
    """
    The main batch processing
    """
    
    
    new_data = initAnalysis(empirical_data_path,plotdata_path,minx,maxx,miny,maxy)
    gvg,tt = loadVariogramFromData(plotdata_path,new_data)
    
    resum,gvgn,resultspd,results = fitGLSRobust(new_data,gvg,num_iterations=50,distance_threshold=1000000)
    
    
    

    #CovMat = buildSpatialStructure(new_data,gvg.model)
    #results,resum = calculateGLS(new_data,CovMat)

    
    logger.info("Writing to file")
    f = open("gls1.txt",'w')
    f.write(resum.as_text())
    f.close()    
    
    logger.info("Finished! Results in: gls1.txt")
    
    return {'dataframe':new_data,'variogram':gvgn,'modelGLS':resum,'results':resultspd,'model_results':results}
    
if __name__ == "__main__":
    empirical_data_path = sys.argv[1]
    plotdata_path = sys.argv[2]
    minx = float(sys.argv[3])
    maxx = float(sys.argv[4])
    miny = float(sys.argv[5])
    maxy = float(sys.argv[6])
    results = main(empirical_data_path,plotdata_path,minx,maxx,miny,maxy)




## Sectionate by coordinates
##minx = -85
##maxx = -80
##miny = 30
##maxy = 35


### PATH information for running in the Biospytial container
#plotdata_path = "/RawDataCSV/idiv_share/plotsClimateData_11092017.csv"
#empirical_data_path = "/apps/external_plugins/spystats/HEC_runs/results/logbiomas_logsppn_res.csv"

## Information for HEC path
## For HEC path in /home/hpc/28/escamill/csv_data/idiv/plotsClimateData_11092017.csv

## For running ALL DATA SET
#-130 -60 25 50
#python /home/hpc/28/escamill/spystats/HEC_runs/fit_fia_logbiomass_logspp_GLS.py /home/hpc/28/escamill/csv_data/idiv/plotsClimateData_11092017.csv /home/hpc/28/escamill/spystats/HEC_runs/results/logbiomas_logsppn_res.csv -130 -60 25 50

## Test small region
# -85 -80 30 35
#python /home/hpc/28/escamill/spystats/HEC_runs/fit_fia_logbiomass_logspp_GLS.py /home/hpc/28/escamill/csv_data/idiv/plotsClimateData_11092017.csv /home/hpc/28/escamill/spystats/HEC_runs/results/logbiomas_logsppn_res.csv -85 -80 30 35  

## Checar la importacion de la libreria spystats



#python fit_fia_sppn.py /RawDataCSV/idiv_share/plotsClimateData_11092017.csv -85 -80 30 35  
## In hec
#python fit_fia_sppn.py /home/hpc/28/escamill/csv_data/idiv/plotsClimateData_11092017.csv -90 -80 30 40 300 

