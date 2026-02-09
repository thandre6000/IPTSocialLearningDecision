
import codecs
import argparse
import glob
import os
import time
import re
from datetime import datetime
import multiprocessing as mp
from multiprocessing import Pool
import pickle
import pandas as pd
import numpy as np
from scipy import optimize
import scipy as sp
from scipy import stats as stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor


def load_workspace(filename):
 with open(filename, 'rb') as f:
  data_loaded = pickle.load(f)
 return data_loaded
data = load_workspace('pklForTheo.pkl')
def genModelPreds_biasOC_d(outcomes,lr,d,p0):
    outcomes=np.array(outcomes)

    ocsDf = np.zeros(outcomes.shape)
    ocsDf[outcomes==-1]=-d
    ocsDf[outcomes==1]=1-d
        
    pred = np.zeros(outcomes.shape)
    pred[:,0] = p0
    for ii in range(outcomes.shape[1]-1):
        pred[:,ii+1] = pred[:,ii] + lr * (ocsDf[:,ii] - pred[:,ii])
    return pred 


def calcActProbsPosNegOC(modelPreds, sftmax, chVals=np.arange(5)+1):
    preds=np.array(modelPreds)
    EVs=[preds*i for i in chVals]
    expEVs_sftmax=[np.exp(np.float64(i)/sftmax) for i in EVs]
    sftmax_denom=np.zeros(expEVs_sftmax[0].shape)
    for i in expEVs_sftmax:
        sftmax_denom+=i
    cols=['ch_{:02d}'.format(i) for i in np.arange(modelPreds.shape[1])+1]
    actProbArr=[pd.DataFrame(i/sftmax_denom, columns=cols) for i in expEVs_sftmax]
    return actProbArr



def calcNegLogLike(choices, actProbs):
    choices.reset_index(inplace=True, drop=True)
    chProbs=choices.copy()
    for i in np.arange(len(actProbs)):
        chProbs[choices==i+1]=actProbs[i][choices==i+1]
    
    chProbs[chProbs==0]=np.nan
    nll=-np.log(np.float64(chProbs))
    np.nan_to_num(nll, copy=False)
    nll=np.sum(nll, axis=1)
    
    return nll, chProbs

def chanceLikelihood(choices, chanceP=0.2):
    choices.reset_index(inplace=True, drop=True)
    chProbs=choices.copy()
    
    chProbs[chProbs!=0]=chanceP
    chProbs[chProbs==0]=np.nan
    nll=-np.log(np.float64(chProbs))
    np.nan_to_num(nll, copy=False)
    nll=np.sum(nll, axis=1)
    
    return nll
#single LR sofented with sigmoid and biasedOC with sigmoid and p0 with no sigmoid, + fixed SM model
def optimizeLR_sft_fixedSM_posNegOC_p0(params, choices, outcomes, smax=1):
    lrp,d, p0=params
    lr=np.exp(lrp)/(np.exp(lrp)+1)
    d=np.exp(d)/(np.exp(d)+1) #if softening d
    p0=(np.exp(p0)/(np.exp(p0)+1))*2-1 #if softening d
    if lr<0 or smax<0 or lr>1:
        nLL=10e30
        return nLL
     
    mod=genModelPreds_biasOC_d(outcomes,lr,d,p0)
    actProbs=calcActProbsPosNegOC(mod, smax)
    nLL, chProbs=calcNegLogLike(choices, actProbs)
    
    return np.sum(nLL)


start_time = datetime.now()

smax=1
lrRange=np.linspace(-2,2,5)
dRange=np.linspace(-2,2,5)
p0Range=np.linspace(-2,2,5)

lrnEstArr=data['lrnEstArr']
lrnOCArr=data['lrnOCArr']
lrnSumm=data['lrnSumm']

fitCount=0;
parser = argparse.ArgumentParser()
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=None)
args = parser.parse_args()

pids_all = list(lrnEstArr.PID.unique())
start = max(0, args.start)
end = len(pids_all) if args.end is None else min(args.end, len(pids_all))
pids = pids_all[start:end]
for PID in pids:
    estDat=lrnEstArr.loc[(lrnEstArr.PID==PID)].transpose().dropna().drop('PID').astype(int)
    estDat.columns=['ch_{:02d}'.format(i) for i in estDat.columns+1]
    ocDat=lrnOCArr.loc[(lrnOCArr.PID==PID)].transpose().dropna().drop('PID').astype(int)*2-1
    lrs_nll=np.inf
    for lrs, ds, p0s in zip(lrRange,dRange, p0Range):
            result_s=sp.optimize.minimize(optimizeLR_sft_fixedSM_posNegOC_p0, (lrs,ds,p0s),(estDat, ocDat),bounds=((-10,10),(-10,10),(-10,10)))
            if result_s.fun < lrs_nll:
                lrs_nll = result_s.fun
                lrs_lr = result_s.x[0]
                lrs_d = result_s.x[1]
                lrs_p0 = result_s.x[2]
 
    lrnSumm.loc[PID,'subjLR_3prm']=np.exp(lrs_lr)/(np.exp(lrs_lr)+1)
    lrnSumm.loc[PID,'subjD_3prm']=np.exp(lrs_d)/(np.exp(lrs_d)+1)
    lrnSumm.loc[PID,'subjp0_3prm']=np.exp(lrs_p0)/(np.exp(lrs_p0)+1)*2 - 1
    lrnSumm.loc[PID,'subjNLL_3prm']=lrs_nll
    chncNLL=np.sum(chanceLikelihood(estDat))
    lrnSumm.loc[PID, 'subjLRTp_3prm']=stats.chi2.sf(2*(chncNLL-lrnSumm.loc[PID,'subjNLL_3prm']),1)

lrnSumm=lrnSumm.loc[:,lrnSumm.columns[~lrnSumm.columns.str.startswith('qa:')].append(lrnSumm.columns[lrnSumm.columns.str.startswith('qa:')])]
    
import os

os.makedirs("results", exist_ok=True)

job_id  = os.environ.get("SLURM_JOB_ID", "local")
task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "0")

tmp_file = f"results/_tmp_{job_id}_{task_id}.csv"
final_file = "results/1LRP0D_TRUST.csv"

# always write a temp file
lrnSumm.loc[pids].to_csv(tmp_file)

# ONLY array task 0 assembles the final file
if task_id == "0":
    import glob
    import pandas as pd

    tmp_files = sorted(glob.glob(f"results/_tmp_{job_id}_*.csv"))
    dfs = [pd.read_csv(f, index_col=0) for f in tmp_files]
    final_df = pd.concat(dfs, axis=0)

    final_df.to_csv(final_file)
    print(f"Wrote final merged file: {final_file}")