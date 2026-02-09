import argparse
import codecs
import sklearn
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
# Example usage:
# loaded_vars = load_workspace('workspace.pkl')
# globals().update(loaded_vars) # This will bring all the variables into the global namespace.
data = load_workspace('pklForTheo.pkl')



def genModelPreds_biasOC_2LR(outcomes,lr_pos,lr_neg,p0):
    outcomes=np.array(outcomes)
    lrsDf = np.zeros(outcomes.shape)
    lrsDf[outcomes==0]=lr_neg
    lrsDf[outcomes==1]=lr_pos
        
    pred = np.zeros(outcomes.shape)
    pred[:,0] = p0
    for ii in range(outcomes.shape[1]-1):
        pred[:,ii+1] = pred[:,ii] + lrsDf[:,ii] * (outcomes[:,ii] - pred[:,ii])
    return pred 


def calcActProbsPosNegOC(modelPreds, sftmax, chVals=np.arange(5)+1):
    preds=np.array(modelPreds)
    EVs = [preds*i + (1-preds)*(-i) for i in chVals]
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



def optimize2LR_sft_fixedSM_p0(params, choices, outcomes, smax=1):
    lr_pos, lr_neg, p0=params
    lr_pos=np.exp(lr_pos)/(np.exp(lr_pos)+1)
    lr_neg=np.exp(lr_neg)/(np.exp(lr_neg)+1)
    #p0 = 1/(1+np.exp(-p0))
    p0=(np.exp(p0)/(np.exp(p0)+1))
    if lr_pos<0 or lr_neg<0 or smax<0 or lr_pos>1 or lr_neg>1:
        nLL=10e30
        return nLL
    mod=genModelPreds_biasOC_2LR(outcomes,lr_pos, lr_neg,p0)
    actProbs=calcActProbsPosNegOC(mod, smax)
    nLL, chProbs=calcNegLogLike(choices, actProbs)
    
    return np.sum(nLL)


def sigmoid(x,a,b,c):
    y=a+b/(1+np.exp(-c*(x-50)))

    return y

start_time = datetime.now()

smax=1
lrRange=np.linspace(-2,2,5)
p0Range=np.linspace(-1, 1, 5)

rwlrnEstArr=data['rwlrnEstArr']
rwlrnOCArr=data['rwlrnOCArr']
rwlrnSumm=data['rwlrnSumm']


fitCount=0;
parser = argparse.ArgumentParser()
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=None)
args = parser.parse_args()

pids_all = list(rwlrnEstArr.PID.unique())
start = max(0, args.start)
end = len(pids_all) if args.end is None else min(args.end, len(pids_all))
pids = pids_all[start:end]

for PID in pids:
    fitCount+=1
    print(f'subj_{fitCount:03.0f} ({datetime.now()-start_time}:0.2f): {PID}')
    rwestDat=rwlrnEstArr.loc[(rwlrnEstArr.PID==PID)].transpose().dropna().drop('PID').astype(int)
    rwestDat.columns=['ch_{:02d}'.format(i) for i in rwestDat.columns+1]
    rwocDat=rwlrnOCArr.loc[(rwlrnOCArr.PID==PID)].transpose().dropna().drop('PID').astype(int)
    lrs_nll=np.inf
    for lrs, p0s in zip(lrRange, p0Range):
        result_s=sp.optimize.minimize(optimize2LR_sft_fixedSM_p0, (lrs,lrs,p0s),(rwestDat, rwocDat), bounds=((-10,10),(-10,10),(-10,10)))
        if result_s.fun < lrs_nll:
            lrs_nll = result_s.fun
            lrs_pos_val = result_s.x[0]
            lrs_neg_val = result_s.x[1]
            lrs_p0 = result_s.x[2]
    rwlrnSumm.loc[PID,'subj2LR_posOC']=np.exp(lrs_pos_val)/(np.exp(lrs_pos_val)+1)
    rwlrnSumm.loc[PID,'subj2LR_negOC']=np.exp(lrs_neg_val)/(np.exp(lrs_neg_val)+1)
    rwlrnSumm.loc[PID,'subj2p0_3prm']= (np.exp(lrs_p0)/(np.exp(lrs_p0)+1))
    rwlrnSumm.loc[PID,'subj2NLL_biasedOC']=lrs_nll
    rwlrnSumm.loc[PID,'subj2LR_biasedOC']=rwlrnSumm.loc[PID,'subj2LR_posOC']-rwlrnSumm.loc[PID,'subj2LR_negOC']
    rwchncNLL=np.sum(chanceLikelihood(rwestDat))
    rwlrnSumm.loc[PID, 'subj2LRTp_biasedOC']=stats.chi2.sf(2*(rwchncNLL-rwlrnSumm.loc[PID,'subj2NLL_biasedOC']),3)
rwlrnSumm=rwlrnSumm.loc[:,rwlrnSumm.columns[~rwlrnSumm.columns.str.startswith('qa:')].append(rwlrnSumm.columns[rwlrnSumm.columns.str.startswith('qa:')])]
print('DONE')

import os

os.makedirs("results", exist_ok=True)

job_id  = os.environ.get("SLURM_JOB_ID", "local")
task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "0")

tmp_file = f"results/_tmp_{job_id}_{task_id}.csv"
final_file = "results/2LRPosNegOCp0NonSocial_LOT.csv"

# always write a temp file
rwlrnSumm.loc[pids].to_csv(tmp_file)

# ONLY array task 0 assembles the final file
if task_id == "0":
    import glob
    import pandas as pd

    tmp_files = sorted(glob.glob(f"results/_tmp_{job_id}_*.csv"))
    dfs = [pd.read_csv(f, index_col=0) for f in tmp_files]
    final_df = pd.concat(dfs, axis=0)

    final_df.to_csv(final_file)
    print(f"Wrote final merged file: {final_file}")