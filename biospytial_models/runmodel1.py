import runner as rn
import redis
import pymc3 as pm
import pickle
import sys

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


if __name__ == '__main__':

    trainkey = sys.argv[1]
    predkey = sys.argv[2]
    outputkey = sys.argv[3]
    inference_method = sys.argv[4]
    ncores = sys.argv[5] 
    niters = sys.argv[6]
    
    redishost='10.42.72.93'
    panthera = redishost
    conn = redis.StrictRedis(host=panthera,password='biospytial.')
    PDF = rn.preparePredictors(rn.loadDataFrameFromRedis(predkey,conn))
    TDF = rn.loadDataFrameFromRedis(trainkey,conn)
    
    formula = 'LUCA ~ Longitude + Latitude + Q("Dist.to.road_m") + Population_m + name'
    
    TM,PM = rn.splitByFormula(formula,TDF,PDF['clean'])
    logger.info("Start modelling inference")
    
    model = rn.ModelSamplingEffort(TM,PM)    
    trace = rn.SampleModel(model,inference_method=inference_method,ncores=ncores,niters=niters)
    pred_sample = rn.SamplePredictions(model,TM,PM)
    
    tracedf = pm.trace_to_dataframe(trace)
    tracedf.to_csv('/storage/users/escamill/presence-only-model/output/trace%s.csv'%outputkey,encoding='utf8')
    pm.save_trace(trace,directory='/storage/users/escamill/presence-only-model/output/rawtrace50')
    pred_sample.to_csv('/storage/users/escamill/presence-only-model/output/pred_cond-%s.csv'%outputkey,encoding='utf8')
        # pred sample is a dictionary
    
    pickle.dump('/storage/users/escamill/presence-only-model/output/pred%s.pickle'%outputkey,pred_sample)
    logger.info("Finished!")
    
    
