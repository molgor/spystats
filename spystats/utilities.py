#!/usr/bin/env python
#-*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

"""
Utilities for reading predictors and training dataset imported from Biospytial suite.
"""


__author__ = "Juan Escamilla Mólgora"
__copyright__ = "Copyright 2018, JEM"
__license__ = "GPL"
__mantainer__ = "Juan"
__email__ ="j.escamilla.molgora@ciencias.unam.mx"


import pandas as pd
import numpy as np
import logging
from os import walk
import traversals.strategies as st
import pymc3 as pm
import matplotlib.pyplot as plt
from raster_api.tools import RasterContainer
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def loadDataset(path_to_dataset):
    """
    Load Datasets (CSV) from a location Path
    """
    logger.info("Reading file %s"%path_to_dataset)
    _files = []
    for (dirpath, dirnames,filenames) in walk(path_to_dataset):
        _files = map(lambda f : dirpath + '/' + f ,filenames)
        ## Read all data
        dataset = map(lambda f : pd.read_csv(f,na_values=["N.A.","NaN","N.A"],encoding='utf8',index_col=0),_files)  
	dataset = map(lambda d : st.toGeoDataFrame(d,xcoord_name='Longitude',ycoord_name='Latitude'),dataset)    
    return dataset


def toGeoDataFrame(pandas_dataframe,xcoord_name='Longitude',ycoord_name='Latitude',srs = 'epsg:4326'):
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
    #import ipdb; ipdb.set_trace()
    data[xcoord_name] = pd.to_numeric(data[xcoord_name])
    data[ycoord_name] = pd.to_numeric(data[ycoord_name])
    data['geometry'] = data.apply(lambda z : Point(z[xcoord_name], z[ycoord_name]), axis=1)
    #data['geometry'] = data.apply(lambda z : Point(z.LON, z.LAT), axis=1)

    new_data = gpd.GeoDataFrame(data)
    new_data.crs = {'init':'epsg:4326'}
    return new_data


### Convert to categorical data
## Example , drop the WWF clasification
def _dummifySerie(serie):
    """
    Parameters :
        serie : A Pandas Serie
    Assigns a column as categorical
    Returns :
        a dummy dataframe

    """
    v = serie.astype('category')
    vc = pd.get_dummies(v,prefix='dum')
    return vc


def dummifyDF(dataframe,column_name,concatenate=False):
    """
    Parameters :
        dataframe (Pandas Dataframe)
        column_name :  (String) the column to be converted to dummy df.
    """
    serie = dataframe[[column_name]]
    dummies = _dummifySerie(serie)
    if concatenate:
        return pd.concat([dataframe,dummies],axis=1)
    else:
        return dummies

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
    ## Remove problematic string columns, so that all of them can be numeric
    #datapred = datapred.drop(['vegname','inegiv5name'],axis=1)
    ## Convert to numeric
    ## Very nice way to convert to numeric 
    #cols = datapred.columns.drop('Unnamed: 0')
    #datapred[cols] = datapred[cols].apply(pd.to_numeric,errors='coerce')
    ## New data set without nas for any other column
    datacube_clean = datapred.dropna()
    ## Convert to geopandas
    #datacube_clean = st.toGeoDataFrame(datacube_clean,xcoord_name='Longitude',ycoord_name='Latitude')
    return {'clean' : datacube_clean, 'full' :datapred}
    #return datacube_clean


## This is for calculating the signal
## This is for calculating the signal
from pymc3.variational.callbacks import CheckParametersConvergence
def FitMyModel(Y,train,predictor):
    #
    with pm.Model() as model:

        ## [R | Y]

        tau = pm.HalfNormal('tau',sd=10)
        sigma = pm.HalfNormal('sigma',sd=10)
        phi = pm.Uniform('phi',0,15)

        Tau = pm.gp.cov.Constant(tau)
        cov = (sigma * pm.gp.cov.Matern32(2,phi,active_dims=[0,1])) + Tau

        ## Parameters for linear predictor
        #b0 = pm.Normal('b0',mu=0,sd=10)
        b = pm.Normal('b',mu=0,sd=10,shape=3)
        mf = pm.gp.mean.Linear(coeffs=[b]) 

        ## The latent function
        gp = pm.gp.Latent(cov_func=cov)
        f = gp.prior("latent_field", X=train[['Longitude','Latitude','DistanceToRoadMex_mean','WorldPopLatam2010_mean','vegid']].values,reparameterize=False)


        ## Other model M2

        beta_y = pm.Normal("betay",mu=0, sd=10,shape=2)

        theta = beta_y[0] + beta_y[1] * train.MaxTemperature_mean.values

        yy = pm.Bernoulli("yy",logit_p=theta,observed=Y.values)


        #y_obs = pm.Bernoulli('y_obs',logit_p=(f*yy),observed=Y.values)


        trace = pm.fit(method='advi', callbacks=[CheckParametersConvergence()],n=15000)    
        #trace = pm.sample(10)
        trace = trace.sample(draws=5000)


        f_star = gp.conditional("f_star", predictor['clean'][['Longitude','Latitude','DistanceToRoadMex','WorldPopLatam2010','vegid']].values)

        pred_samples = pm.sample_ppc(trace, vars=[f_star], samples=100)
        return pred_samples
   


def plotThings(pred_samples,datapred,rastertemplate):
    preds = pd.DataFrame(pred_samples['f_star']).transpose()
    import scipy.special as sp
    alpha = sp.logit(0.5)
    mean_sample = preds.mean(axis=1)
    q_025 = preds.quantile(0.025,axis=1)
    q_975 = preds.quantile(0.975,axis=1)
    prob_gt05 = preds.apply(lambda row : float(sum(row > alpha))/100,axis=1)
    surface_data = pd.DataFrame({'mean_sample' : mean_sample, 'q_025':q_025,'q_975':q_975,'prob_gt05':prob_gt05})

    #preds['idx'] = data_star.index.values
    surface_data['idx'] = datapred['clean'].index.values
    predictions = datapred['full'].merge(surface_data,how='left',left_index=True,right_on='idx',suffixes=('_obs','_pred'))
    predicted_data = predictions.mean_sample.values
    # Raster Container
    ql_presences_of_something = RasterContainer(predictions.q_025.values,use_metadata_from=rastertemplate.rasterdata,exponential_fam="Bernoulli")
    ql_presences_of_something.display_field(band=2,origin='Lower',cmap=plt.cm.RdBu,interpolation='None',title="Quantiles 0.025")

    mean_presences_of_something = RasterContainer(predicted_data,use_metadata_from=rastertemplate.rasterdata,exponential_fam="Bernoulli")
    mean_presences_of_something.display_field(band=2,origin='Lower',cmap=plt.cm.RdBu,interpolation='None',title="mean probability")

    qh_presences_of_something = RasterContainer(predictions.q_975.values,use_metadata_from=rastertemplate.rasterdata,exponential_fam="Bernoulli")
    qh_presences_of_something.display_field(band=2,origin='Lower',cmap=plt.cm.RdBu,interpolation='None',title="quantile 0.097")

    ## Probability of the probaaility of presences bigger than 0.5
    prob5 = RasterContainer(predictions.prob_gt05.values,use_metadata_from=rastertemplate.rasterdata,exponential_fam="Bernoulli")
    prob5.display_field(band=2,origin='Lower',cmap=plt.cm.RdBu,interpolation='None',title="probability of exceding more than 0.5")

    return None
