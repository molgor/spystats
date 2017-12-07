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
import logging

## Use the ggplot style
plt.style.use('ggplot')

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# create a file handler
#handler = logging.FileHandler('batchprocess.log')
#handler.setLevel(logging.INFO)


# create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#handler.setFormatter(formatter)


# add the handlers to the logger
#logger.addHandler(handler)




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
model = smf.ols(formula='logBiomass ~ logSppN',data=new_data)
results = model.fit()
param_model = results.params
results.summary()
new_data['residuals'] = results.resid

logger.info("Removing possible duplicates")
new_data = new_data.drop_duplicates(subset=['newLon','newLat'])




## Read the empirical variogram
logger.info("Reading the empirical Variogram file")
thrs_dist = 100000


## Change here with appropriate path for file
empirical_semivariance_log_log = "../HEC_runs/results/logbiomas_logsppn_res.csv"

#### here put the hec calculated 
emp_var_log_log = pd.read_csv(empirical_semivariance_log_log)
gvg = tools.Variogram(new_data,'logBiomass',using_distance_threshold=thrs_dist)
gvg.envelope = emp_var_log_log
gvg.empirical = emp_var_log_log.variogram
gvg.lags = emp_var_log_log.lags
emp_var_log_log = emp_var_log_log.dropna()
vdata = gvg.envelope.dropna()
#gvg.plot(refresh=False,legend=False,percentage_trunked=20)
#plt.title("Semivariogram of residuals $log(Biomass) ~ log(SppR)$")



