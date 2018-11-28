#import traversals.strategies as st
#from os import walk
import matplotlib.pyplot as plt
import pandas as pd
import itertools as it
import numpy as np
import pymc3 as pm
import sys


from pymc3.variational.callbacks import CheckParametersConvergence
def FitMyModel(trainDM,PredDM):
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
        x_index = TXdf.columns.get_loc("Longitude")
        y_index = TXdf.columns.get_loc("Latitude")
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
        trace = pm.sample(15,init='adapt_diag')
        #trace = trace.sample(draws=5000)
        # Remove any column that doesnt appear in the training data
        ValidPreds = PredDM[TXdf.columns]
        PredX = ValidPreds.values
        f_star = gp.conditional("f_star", PredX)
        pred_samples = pm.sample_ppc(trace, vars=[f_star], samples=100)
        return pred_samples,trace






sys.path.append('/home/hpc/28/escamill/spystats')

from spystats import utilities as ut

train_path = '/storage/users/escamill/presence-only-model/input/train'
train_dataset = ut.loadDataset(train_path)

#train_path = '/outputs/presence_only_models/data/root'
#train_dataset = ut.loadDataset(train_path)
## Predictors
pred_path = '/storage/users/escamill/presence-only-model/input/pred'
pred_dataset = ut.loadDataset(pred_path)
### PATCH, the thing is taking backwards the order of the lists of files, because of the name
#pred_dataset.reverse()
prediction_dataset_dic= map(lambda p : ut.preparePredictors(p),pred_dataset)





i = 4

datatrain = train_dataset[i]
#Y = datatrain.Burseraceae
#Y = datatrain.Burseraceae
datapred = prediction_dataset_dic[i]


## Assign categorical values
datatrain.name = datatrain.name.astype('category')
datapred['full'].name = datapred['full'].name.astype('category')
datapred['clean'].name = datapred['clean'].name.astype('category')

from patsy import dmatrices,dmatrix
TM = dmatrices('LUCA ~ Longitude + Latitude + Q("Dist.to.road_m") + Population_m ',datatrain,return_type="dataframe")
#TM = dmatrices('Burseraceae ~ Longitude + Latitude + DistanceToRoadMex_mean + WorldPopLatam2010_mean + inegiv5name',datatrain)

PM = dmatrix('Longitude + Latitude + Q("Dist.to.road_m") + Population_m',datapred['clean'],return_type='dataframe')
#PM = dmatrix('Longitude + Latitude + Q("Dist.to.road_m") + Population_m + name',datapred['clean'])




