#!/usr/bin/env python
#-*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

"""
Runner file to be used in HEC
"""


__author__ = "Juan Escamilla Mólgora"
__copyright__ = "Copyright 2018, JEM"
__license__ = "GPL"
__mantainer__ = "Juan"
__email__ ="j.escamilla.molgora@ciencias.unam.mx"

#import traversals.strategies as st
#from os import walk
import matplotlib.pyplot as plt
import pandas as pd
import itertools as it
import numpy as np
import pymc3 as pm
import sys
import redis
import pickle
import logging
from patsy import dmatrices,dmatrix
#import ipdb; ipdb.set_trace()

#from spystats import utilities as ut

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def ModelSamplingEffort(trainDM,PredDM):
    """
    Sampling effort model
    parameters:
        trainDM : (duple of dataframes). loc 0 is the lhs term of the equation.
        PredDM : dataframe
        pymc_model : A Pymc3 model object

    notes:
        Should be the output of the patsy.dmatrices and dmatrix respectively
    """
    with pm.Model() as model:

        # partition dataframes df
        Ydf = trainDM[0]
        TXdf = trainDM[1]

        PXdf = PredDM

        ## Parameters for linear predictor
        #b0 = pm.Normal('b0',mu=0,sd=10)
        #dum_names = filter(lambda col : str(col).startswith('inegiv5name'),TXdf)
        #dumsdf = TXdf[dum_names]
        #dumshape = dumscols.shape
        #coordsdf = TXdf[['Longitude','Latitude']]

        # Create vectors for dumi vars 
        #drvs = map(lambda col : pm.Normal(col,mu=0,sd=1.5),dum_names)
        ## Create theano vector
        dimX = len(TXdf.columns)
        b = pm.Normal('b',mu=0,sd=1.5,shape=dimX)
        #mk = pm.math.matrix_dot(TXdf.values,b.transpose())



        ## The latent function
        x_index = TXdf.columns.get_loc(b"Longitude")
        y_index = TXdf.columns.get_loc(b"Latitude")

        ## Building the covariance structure
        tau = pm.HalfNormal('tau',sd=10)
        sigma = pm.HalfNormal('sigma',sd=10)
        #phi = pm.Uniform('phi',0,15)
        phi = pm.HalfNormal('phi',sd=6)
        Tau = pm.gp.cov.Constant(tau)
        cov = (sigma * pm.gp.cov.Matern32(2,phi,active_dims=[x_index,y_index])) + Tau

        mean_f = pm.gp.mean.Linear(coeffs=b)

        gp = pm.gp.Latent(mean_func=mean_f,cov_func=cov)

        f = gp.prior("latent_field", X=TXdf.values,reparameterize=False)


        yy = pm.Bernoulli("yy",logit_p=f,observed=Ydf.values)



        #trace = pm.fit(method='advi', callbacks=[CheckParametersConvergence()],n=15000)    
        trace = pm.sample(150,init='adapt_diag')
        #trace = trace.sample(draws=5000)

        # Remove any column that doesnt appear in the training data
        ValidPreds = PredDM[TXdf.columns]
        PredX = ValidPreds.values

        f_star = gp.conditional("f_star", PredX)

        pred_samples = pm.sample_ppc(trace, vars=[f_star], samples=100)

    return pred_samples,trace

	        #trace = pm.fit(method='advi', callbacks=[CheckParametersConvergence()],n=15000)    


def loadDataFrameFromRedis(keyname,redis_connection):
    """
    Loads a Pandas dataframe stored in redis given a key
    """
    logger.info("Loading data from RedisDB")
    return pickle.loads(redis_connection.get(keyname))


def preparePredictors(pred_dataset):
    """
    Prepares the predictor datasets.
    Parameters : 
        pred_dataset : A list of dataframes obtained from the predictor function 
        i.e. compileDataCube (for the moment hold in notebook, obtaining_predictors)
    Returns :
        list of geopandas dataframes.
    
    """
    datapred = pred_dataset
    datapred = datapred.replace("N.A",np.nan)
    ## Remove NAs from coordinates (necessary, given the hereogeneous datasource (included in worldpop and elevation)
    datapred.dropna(subset=['Longitude','Latitude'],inplace=True)
    datacube_clean = datapred.dropna()
    return {'clean' : datacube_clean, 'full' :datapred}


def splitByFormula(formula,traindf,predictordf):
    """
    Splits the dataframe into separate dataframes given a formula
    Parameters :
        formula : (String) in patsy language.
        traindf : (pandas.DataFrame) The dataframe for training.
        predictordf : (pandas.DataFrame) the dataframe for prediction
    """
    logger.info("Creating Design matrices from the formula %s"%formula)
    lhs,rhs = formula.split('~')
    TM = dmatrices(formula ,traindf,return_type="dataframe")
    PM = dmatrix(rhs,predictordf,return_type='dataframe')
    return (TM,PM)

def main():
    panthera = '10.42.72.93'
    conn = redis.StrictRedis(host=panthera,password='biospytial.')
    predkey = 'p-50x50-guerrero-4'
    trainkey = 't-luca-guerrero-4'
    PDF = preparePredictors(loadDataFrameFromRedis(predkey,conn))
    TDF = loadDataFrameFromRedis(trainkey,conn)

    formula = 'LUCA ~ Longitude + Latitude + Q("Dist.to.road_m") + Population_m + name'

    TM,PM = splitByFormula(formula,TDF,PDF['clean'])
    logger.info("Start modelling inference")

    modelo = ModelSamplingEffort(TM,PM)

    conn.set('test-model',pickle.dumps(modelo))
    logger.info("Finish!")

if __name__ == '__main__':
    
    main()

