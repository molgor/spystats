%history
fia_data_path = "/home/hpc/28/escamill/csv_data/idiv/FIA_Plots_Biomass_11092017.csv"
plotdata_path = '/home/hpc/28/escamill/spystats/HEC_runs/results/logbiomas_logsppn_res.csv'
empirical_data_path = fia_data_path
minx = -130
maxx = -60
miny = 24
maxy = 50
new_data = initAnalysis(empirical_data_path,plotdata_path,minx,maxx,miny,maxy)
from HEC_runs.fit_fia_logbiomass_logspp_GLS import prepareDataFrame,loadVariogramFromData,buildSpatialStructure, calculateGLS, initAnalysis,fitGLSRobust
pwd
cd ..
from HEC_runs.fit_fia_logbiomass_logspp_GLS import prepareDataFrame,loadVariogramFromData,buildSpatialStructure, calculateGLS, initAnalysis,fitGLSRobust
ls
cd spystats/
ls
from HEC_runs.fit_fia_logbiomass_logspp_GLS import prepareDataFrame,loadVariogramFromData,buildSpatialStructure, calculateGLS, initAnalysis,fitGLSRobust
from fit_fia_logbiomass_logspp_GLS import prepareDataFrame,loadVariogramFromData,buildSpatialStructure, calculateGLS, initAnalysis,fitGLSRobust
from HEC_runs.fit_fia_logbiomass_logspp_GLS import prepareDataFrame,loadVariogramFromData,buildSpatialStructure, calculateGLS, initAnalysis,fitGLSRobust
cd ..
from HEC_runs.fit_fia_logbiomass_logspp_GLS import prepareDataFrame,loadVariogramFromData,buildSpatialStructure, calculateGLS, initAnalysis,fitGLSRobust
cd spystats/
ls
%history
ls
import tools
cd ..
from HEC_runs.fit_fia_logbiomass_logspp_GLS import prepareDataFrame,loadVariogramFromData,buildSpatialStructure, calculateGLS, initAnalysis,fitGLSRobust
new_data = initAnalysis(empirical_data_path,plotdata_path,minx,maxx,miny,maxy)
new_data.columns
new_data.length
CovMat = np.load("/storage/users/escamill/covmatfia.npy")
import numpy as np
CovMat = np.load("/storage/users/escamill/covmatfia.npy")
ls /storage/users/escamill/
CovMat = np.load("/storage/users/escamill/final_covariance_matrix_glsFIA.npy")
results,resum = calculateGLS(new_data,CovMat)
results
resum
%hist -f replay_analysis.py
