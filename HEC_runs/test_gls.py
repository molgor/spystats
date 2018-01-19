## Script for testing GLS and Positive Definite Matrix
import numpy as np
CovMat = np.load("/storage/users/escamill/covmatfia.npy")
x = np.linalg.cholesky(CovMat)
import sys
import tools
import geopandas as gpd
from HEC_runs.fit_fia_logbiomass_logspp_GLS import prepareDataFrame, createVariogram, buildSpatialStructure,calculateGLS, bundleToGLS, fitLinearLogLogModel
new_data = prepareDataFrame("/home/hpc/28/escamill/csv_data/idiv/plotsClimateData_11092017.csv")
gvg,tt = createVariogram("/home/hpc/28/escamill/spystats/HEC_runs/results/logbiomas_logsppn_res.csv",new_data)

#import statsmodels.regression.linear_model as lm
#model1 = lm.GLS.from_formula(formula='logBiomass ~ logSppN',data=new_data,sigma=CovMat)
checkMat = lambda M : np.linalg.cholesky(np.linalg.pinv(M))
