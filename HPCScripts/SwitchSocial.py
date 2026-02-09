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




def genModelPreds_biasOC_2LR_SW(data, lr_pre, lr_post, p0, d):
    data = data.astype(int)*2-1
    outcome_rows = [r for r in data.index if not r.startswith('sw_')]
    switch_rows = [r for r in data.index if r.startswith('sw_')]


    outcomes = data.loc[outcome_rows].to_numpy()
    switch_phase = data.loc[switch_rows].to_numpy()
    ocsDf = np.zeros(outcomes.shape)
    ocsDf[outcomes == -1] = -d
    ocsDf[outcomes == 1] = 1 - d
    
    lrsDf = np.zeros(outcomes.shape)
    lrsDf[switch_phase == -1] = lr_pre
    lrsDf[switch_phase == 1] = lr_post
        
    pred = np.zeros(outcomes.shape)
    pred[:,0] = p0
    for ii in range(outcomes.shape[1]-1):
        pred[:,ii+1] = pred[:,ii] + lrsDf[:,ii] * (ocsDf[:,ii] - pred[:,ii])
    return pred


def calcActProbs(modelPreds, sftmax, chVals=np.arange(5)+1):
    preds=np.array(modelPreds)
    EVs=[preds*i+(1-preds)*-i for i in chVals]
    expEVs_sftmax=[np.exp(np.float64(i)/sftmax) for i in EVs]
    sftmax_denom=np.zeros(expEVs_sftmax[0].shape)
    for i in expEVs_sftmax:
        sftmax_denom+=i
    cols=['ch_{:02d}'.format(i) for i in np.arange(modelPreds.shape[1])+1]
    actProbArr=[pd.DataFrame(i/sftmax_denom, columns=cols) for i in expEVs_sftmax]
    
    return actProbArr

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

def genChoices(actProbArr):
    probCh=[np.zeros(actProbArr[0].shape)]
    choices=np.random.random_sample(actProbArr[0].shape)
    for i in range(len(actProbArr)):
        probCh.append(probCh[-1]+actProbArr[i])
        choices[choices<probCh[-1]]=i+1
    
    return choices#, probCh

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

def genBetaParams(mu=0.3, sd=0.05):
    alpha=((1-mu)/sd**2 - 1/mu )*mu**2
    beta=alpha*(1/mu-1)
    
    return alpha, beta

def betaMuSd(mu=0.3, sd=0.05, N=1):
    alpha=((1-mu)/sd**2 - 1/mu )*mu**2
    beta=alpha*(1/mu-1)
    



#pre and post switch LRS
def optimize2LR_sft_fixedSM_posNegOC_p0_switch(params, choices, outcomes, smax=1):
    lr_pre, lr_post, p0, d =params
    lr_pre=np.exp(lr_pre)/(np.exp(lr_pre)+1)
    lr_post=np.exp(lr_post)/(np.exp(lr_post)+1)
    d=np.exp(d)/(np.exp(d)+1) #if softening d
    p0=(np.exp(p0)/(np.exp(p0)+1))*2-1 #if softening p0
    if lr_post<0 or lr_pre<0 or smax<0 or lr_post>1 or lr_pre>1:
        nLL=10e30
        return nLL
     
    mod=genModelPreds_biasOC_2LR_SW(outcomes,lr_pre,lr_post,p0,d)
    actProbs=calcActProbsPosNegOC(mod, smax)
    nLL, chProbs=calcNegLogLike(choices, actProbs)
    
    return np.sum(nLL)


def bootstrap_correlation_ci(series1, series2, n_bootstraps=10000, ci=95):
    indices = np.arange(len(series1))
    correlations = []
    
    for _ in range(n_bootstraps):
        boot_indices = np.random.choice(indices, size=len(indices), replace=True)
        boot_series1 = series1[boot_indices]
        boot_series2 = series2[boot_indices]
        r, _ = stats.pearsonr(boot_series1, boot_series2)
        correlations.append(r)
    
    lower = np.percentile(correlations, (100 - ci) / 2)
    upper = np.percentile(correlations, 100 - (100 - ci) / 2)
    return lower, upper

# And update your function definition to:
def bootstrap_mean_ci(series, n_bootstraps=1000, ci=95):
    means = []
    indices = np.arange(len(series))
    for _ in range(n_bootstraps):
        boot_indices = np.random.choice(indices, size=len(indices), replace=True)
        boot_series = series.iloc[boot_indices]
        means.append(np.mean(boot_series))
    lower = np.percentile(means, (100 - ci) / 2)
    upper = np.percentile(means, 100 - (100 - ci) / 2)
    return lower, upper

def sigmoid(x,a,b,c):
    y=a+b/(1+np.exp(-c*(x-50)))

    return y

def add_switch_phase_trust(group):
    # The switch point should be the same for  rows in the group
    switch_point = group['agentswpt'].iloc[0]
    # Create the 'switch_phase' column
    # 0 for pre-switch, 1 for post-switch
    group['switch_phase'] = (group['aground'] > switch_point).astype(int)
    return group

## Now the trust data
lrnEstArr=data['lrnEstArr']
lrnOCArr=data['lrnOCArr']
lrnSumm=data['lrnSumm']
# do data wrangling to add the pre post switch parts to the outcome arrays
lrnOCArr = lrnOCArr.reset_index()
lrnOCArr = lrnOCArr.rename(columns={'index': 'aground'})
lrnOCArr['aground'] = lrnOCArr['aground']+1
## do data wrangling to add the pre post switch parts to the outcome arrays
subj_ids = data['subjs']



switch_phase_dfs_trust = []
for i, subject_data in enumerate(data['lrnDat']):
    # Create a DataFrame for the current participant
    df = pd.DataFrame(subject_data)
    df = df[df['trialtype'] == 'decision']
    # Apply the function to add the 'switch_phase' column
    df['agenttype'] = df['agenttype'].astype('category')
    processed_df = df.groupby('agenttype', group_keys=False).apply(add_switch_phase_trust)
    outcome_mapping = {'Won': 1, 'Lost': 0}
    processed_df['outcome'] = processed_df['outcome'].map(outcome_mapping)
    # Keep essential columns for matching
    switch_phase_dfs_trust.append(processed_df[['subject_id', 'agenttype', 'aground', 'outcome', 'switch_phase']])

# 2. Concatenate all switch phase dataframes
switch_phase_long_df_trust = pd.concat(switch_phase_dfs_trust, ignore_index=True)
# 3. Pivot the data from long to wide format
switch_phase_long_df_trust['agent_trial_col'] = switch_phase_long_df_trust['agenttype'].astype(str)

switch_phase_wide_df_trust = switch_phase_long_df_trust.pivot_table(
    index=['aground','subject_id'], 
    columns=['agenttype'],
    values='switch_phase'
).reset_index().set_index('aground')
cols = switch_phase_wide_df_trust.columns
rename_map = {
    col: f"sw_{col}"
    for col in cols[1:]      # skip first two columns
}

switch_phase_wide_df_trust.rename(columns=rename_map, inplace=True)

# Rename the columns of the pivoted dataframe to avoid conflicts if necessary

switch_phase_wide_df_trust = switch_phase_wide_df_trust.rename(columns={'subject_id': 'PID'})
lrnOCArr = pd.merge(
    lrnOCArr, 
    switch_phase_wide_df_trust, 
    on=['PID', 'aground'], 
    how='left'
)

lrnOCArr.set_index(lrnOCArr.columns[0], inplace=True)
lrnOCArr.index.name = None


cols_to_convert_lrn = lrnOCArr.columns[9:]
lrnOCArr[cols_to_convert_lrn] = lrnOCArr[cols_to_convert_lrn]

# FIT THE LOTTERY and TRUST DATA TOGETHER WITH DUAL LRs for switch, bias, and initialEstimate

start_time = datetime.now()

smax=1
lrRange=np.linspace(-2,2,5)
drange=np.linspace(-2,2,5)
p0Range=np.linspace(-2,2,5)



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
    fitCount+=1
    print(f'subj_{fitCount:03.0f} ({datetime.now()-start_time}:0.2f): {PID}')
    # print('subj: {}'.format(PID))
    estDat=lrnEstArr.loc[(lrnEstArr.PID==PID)].transpose().dropna().drop('PID').astype(int)
    estDat.columns=['ch_{:02d}'.format(i) for i in estDat.columns+1]
    ocDat=lrnOCArr.loc[(lrnOCArr.PID==PID)].transpose().dropna().drop('PID').astype(int)
    lrs_nll=np.inf
    for lrs, p0, d in zip(lrRange,p0Range,drange):
        result_s=sp.optimize.minimize(optimize2LR_sft_fixedSM_posNegOC_p0_switch, (lrs,lrs,p0,d),(estDat, ocDat), bounds=((-10,10),(-10,10),(-10,10),(-10,10)))
        if result_s.fun < lrs_nll:
            lrs_nll = result_s.fun
            lrs_pre_val = result_s.x[0]
            lrs_post_val = result_s.x[1]
            lrs_p0 = result_s.x[2]
            lrs_d = result_s.x[3]

    lrnSumm.loc[PID,'subj2LR_pre']=np.exp(lrs_pre_val)/(np.exp(lrs_pre_val)+1)
    lrnSumm.loc[PID,'subj2LR_post']=np.exp(lrs_post_val)/(np.exp(lrs_post_val)+1)
    lrnSumm.loc[PID,'subj2p0_4prm']=np.exp(lrs_p0)/(np.exp(lrs_p0)+1)*2 - 1
    lrnSumm.loc[PID,'subj2D_4prm']=np.exp(lrs_d)/(np.exp(lrs_d)+1)
    lrnSumm.loc[PID,'subj2NLL_biasedOC']=lrs_nll
    chncNLL=np.sum(chanceLikelihood(estDat))  
    lrnSumm.loc[PID, 'subj2LRTp_biasedOC']=stats.chi2.sf(2*(chncNLL-lrnSumm.loc[PID,'subj2NLL_biasedOC']),1)
lrnSumm=lrnSumm.loc[:,lrnSumm.columns[~lrnSumm.columns.str.startswith('qa:')].append(lrnSumm.columns[lrnSumm.columns.str.startswith('qa:')])]


import os

os.makedirs("results", exist_ok=True)

job_id  = os.environ.get("SLURM_JOB_ID", "local")
task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "0")

tmp_file = f"results/_tmp_{job_id}_{task_id}.csv"
final_file = "results/BiasOut1LR_TRUST.csv"

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
