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


def my_pearsonr(x,y):
    return stats.pearsonr(x,y)[0]
# Generate RL model predictions given outcomes, a learning rate and an initial prediction
#assumes outcomes is a DataFrame, cols=number of trials, rows = number of learning instances
# returns predictions as an np.array with the same shape as outcomes.

def genModelPreds(outcomes,lr,p0):
    pred = np.zeros(outcomes.shape)
    pred[:,0] = p0
    for ii in range(outcomes.shape[1]-1):
        pred[:,ii+1] = pred[:,ii] + lr * (np.array(outcomes.iloc[:,ii]) - pred[:,ii])
    return pred           

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
    lrsDf[switch_phase == 0] = lr_pre
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
    
    return np.random.beta(alpha, beta,N)


#single LR sofented with sigmoid and single SM model
def optimizeLR_sft(params, choices, outcomes):
    lrp,sm=params
    lr=np.exp(lrp)/(np.exp(lrp)+1)
    if np.any(np.array((lr,sm))<0) or lr>1:
        nLL=10e30
        return nLL
     
    mod=genModelPreds(outcomes,lr,initPred)
    actProbs=calcActProbs(mod, sm)
    nLL, chProbs=calcNegLogLike(choices, actProbs)
    
    return np.sum(nLL)


#single LR sofented with sigmoid and fixed SM model
def optimizeLR_sft_fixedSM(params, choices, outcomes, smax=1, initPred=.5):
    lrp=params
    lr=np.exp(lrp)/(np.exp(lrp)+1)
    if lr<0 or smax<0 or lr>1:
        nLL=10e30
        return nLL
     
    mod=genModelPreds(outcomes,lr,initPred)
    actProbs=calcActProbs(mod, smax)
    nLL, chProbs=calcNegLogLike(choices, actProbs)
    
    return np.sum(nLL)

#single LR sofented with sigmoid and fixed SM model
def optimizeLR_sft_fixedSM_posNegOC(params, choices, outcomes, smax=1, initPred=0):
    lrp,d=params
    lr=np.exp(lrp)/(np.exp(lrp)+1)
    d=np.exp(d)/(np.exp(d)+1) #if softening d
    if lr<0 or smax<0 or lr>1:
        nLL=10e30
        return nLL
     
    mod=genModelPreds_biasOC_d(outcomes,lr,d,initPred)
    actProbs=calcActProbsPosNegOC(mod, smax)
    nLL, chProbs=calcNegLogLike(choices, actProbs)
    
    return np.sum(nLL)

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


#pre and post switch LRS
def optimize2LR_sft_fixedSM_posNegOC_p0_switch(params, choices, outcomes, smax=1):
    lr_pre, lr_post, d, p0 =params
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



#double LR sofented with sigmoid and fixed SM model
def optimize2LR_sft_fixedSM(params, choices, outcomes, smax=1, initPred=.5):
    lr_pos, lr_neg=params
    lr_pos=np.exp(lr_pos)/(np.exp(lr_pos)+1)
    lr_neg=np.exp(lr_neg)/(np.exp(lr_neg)+1)
    if lr_pos<0 or lr_neg<0 or smax<0 or lr_pos>1 or lr_neg>1:
        nLL=10e30
        return nLL
     
    mod=genModelPreds_biasOC_2LR(outcomes,lr_pos, lr_neg,initPred)
    actProbs=calcActProbs(mod, smax)
    nLL, chProbs=calcNegLogLike(choices, actProbs)
    
    return np.sum(nLL)


def optimize2LR_sft_fixedSM_p0(params, choices, outcomes, smax=1):
    lr_pos, lr_neg, p0=params
    lr_pos=np.exp(lr_pos)/(np.exp(lr_pos)+1)
    lr_neg=np.exp(lr_neg)/(np.exp(lr_neg)+1)
    p0=(np.exp(p0)/(np.exp(p0)+1))*2-1 #p0 with no sigmoid? confused abt this
    if lr_pos<0 or lr_neg<0 or smax<0 or lr_pos>1 or lr_neg>1:
        nLL=10e30
        return nLL
     
    mod=genModelPreds_biasOC_2LR(outcomes,lr_pos, lr_neg,p0)
    actProbs=calcActProbs(mod, smax)
    nLL, chProbs=calcNegLogLike(choices, actProbs)
    
    return np.sum(nLL)
# assuming shared variance only affects LR
# COV_1LR: Participant data will be fit with a single learning rate intercept (alpha_INT), one or more covariate parameters (e.g.,
# alpha_COV_PC_BPD_comm_PTSD and alpha_COV_PC_BPD_uniq_PTSD), and a single inverse temperature (beta), all at the group-level. Expected
# probability of partner sharing or lottery generating a win (p(0)) will be initialized to 0.5.
def optimize_1lr_PCA_2W(params, choices, outcomes, pca_vec, smax=1, initPred=0.5):
    """
    params = [alpha_INT, alpha_COV_pc1, alpha_COV_pc2]
      - all are group-level coefficients on the *logit* scale
    pca_vec = [z_pc1, z_pc2] for THIS participant
    """
    alpha_INT, w1, w2 = params

    # the group level intercept + covariate contributions 
    x_alpha = alpha_INT + w1 * pca_vec[0] + w2 * pca_vec[1]

    # map to (0,1)
    lr = np.exp(x_alpha) / (np.exp(x_alpha) + 1)

    # generate predictions 
    mod = genModelPreds(outcomes, lr, initPred)

    # softmax and NLL using your helpers
    actProbs = calcActProbs(mod, smax)
    nLL, _ = calcNegLogLike(choices, actProbs)

    return np.sum(nLL)

#with only the shared PC 
def optimize_1lr_PCA_1W(params, choices, outcomes, pca_vec, smax=1, initPred=0.5):
    """
    params = [alpha_INT, alpha_COV_pc1, alpha_COV_pc2]
      - all are group-level coefficients on the *logit* scale
    pca_vec = [z_pc1, z_pc2] for THIS participant
    """
    alpha_INT, w1 = params
    # predictor for learning rate
    x_alpha = alpha_INT + w1 * pca_vec[0]
    # map to (0,1)
    lr = np.exp(x_alpha) / (np.exp(x_alpha) + 1)
    # generate predictions 
    mod = genModelPreds(outcomes, lr, initPred)
    # softmax and NLL using your helpers
    actProbs = calcActProbs(mod, smax)
    nLL, _ = calcNegLogLike(choices, actProbs)
    return np.sum(nLL)

# PTP_LR+COV_p0: For each participant, data will be fit with a single participant-level learning rate (alpha(ptp)) at the participant level, as well as a
# single p0 intercept (p0_INT), one or more covariate parameters (e.g., p0_COV_PC_BPD_comm_PTSD and p0_COV_PC_BPD_uniq_PTSD), and a single
# inverse temperature (beta), all at the group-level.
def optimize_1lr_p0_PCA_2W(params, choices, outcomes, pca_vec, smax=1):
    """
    params = [alpha_INT, alpha_COV_pc1, alpha_COV_pc2]
      - all are group-level coefficients on the *logit* scale
    pca_vec = [z_pc1, z_pc2] for THIS participant
    """
    w1, w2, p0_INT = params

    # latent (real-line) predictor for learning rate
    x_p0 = p0_INT + w1 * pca_vec[0] + w2 * pca_vec[1]

    x_p0=(np.exp(p0)/(np.exp(p0)+1))*2-1
    # generate predictions 
    mod = genModelPreds(outcomes, x_p0)

    # softmax and NLL using your helpers
    actProbs = calcActProbs(mod, smax)
    nLL, _ = calcNegLogLike(choices, actProbs)

    return np.sum(nLL)

# COV_2LR MODEL: Participant data will be fit with two group-level learning rate intercepts (alpha_OC+_INT, alpha_OC-_INT), each of which will be
# paired with one or more group-level covariate parameters (e.g., alpha_OC+_COV_PC_BPD_comm_PTSD and alpha_OC+_COV_PC_BPD_uniq_PTSD),
# and a single group-level inverse temperature (beta). Expected probability of partner sharing or lottery generating a win (p(0)) will be initialized to
# 0.5..

def optimize_2lr_PCA_1W(params, choices, outcomes, pca_vec, smax=1, initPred=0.5):
    """
    params = [alpha_INT, alpha_COV_pc1, alpha_COV_pc2]
      - all are group-level coefficients on the *logit* scale
    pca_vec = [z_pc1, z_pc2] for THIS participant
    """
    alpha_OC_POS_INT, alpha_OC_NEG_INT, w1 = params

    # the group level intercept + covariate contributions 
    x_alpha_1 = (alpha_OC_POS_INT+ (w1 * pca_vec[0])) 
    x_alpha_2 = (alpha_OC_NEG_INT+ (w1 * pca_vec[0]))
    # map to (0,1)
    lr_pos = np.exp(x_alpha_1) / (np.exp(x_alpha_1) + 1)
    lr_neg = np.exp(x_alpha_2) / (np.exp(x_alpha_2) + 1)
    # generate predictions 
    mod = genModelPreds_biasOC_2LR(outcomes, lr_pos, lr_neg, initPred)

    # softmax and NLL using your helpers
    actProbs = calcActProbs(mod, smax)
    nLL, _ = calcNegLogLike(choices, actProbs)

    return np.sum(nLL)
# COV_2LR+PTP_p0 MODEL: Participant data will be fit with two group-level learning rate intercepts (alpha_OC+_INT, alpha_OC-_INT), each of which
# will be paired with one or more group-level covariate parameters (e.g., alpha_OC+_COV_PC_BPD_comm_PTSD and
# alpha_OC+_COV_PC_BPD_uniq_PTSD), a single participant-level initial expectation (p0(ptp)), and a single group-level inverse temperature (beta).
def optimize_2lr_p0_PCA_1W(params, choices, outcomes, pca_vec, smax=1):
    """
    params = [alpha_INT, alpha_COV_pc1, alpha_COV_pc2]
      - all are group-level coefficients on the *logit* scale
    pca_vec = [z_pc1, z_pc2] for THIS participant
    """
    alpha_OC_POS_INT, alpha_OC_NEG_INT, w1, p0 = params

    # the group level intercept + covariate contributions 
    x_alpha_1 = (alpha_OC_POS_INT+ (w1 * pca_vec[0])) 
    x_alpha_2 = (alpha_OC_NEG_INT+ (w1 * pca_vec[0]))
    # map to (0,1)
    lr_pos = np.exp(x_alpha_1) / (np.exp(x_alpha_1) + 1)
    lr_neg = np.exp(x_alpha_2) / (np.exp(x_alpha_2) + 1)

    p0=(np.exp(p0)/(np.exp(p0)+1))*2-1

    # generate predictions 
    mod = genModelPreds_biasOC_2LR(outcomes, lr_pos, lr_neg, p0)

    # softmax and NLL using your helpers
    actProbs = calcActProbs(mod, smax)
    nLL, _ = calcNegLogLike(choices, actProbs)

    return np.sum(nLL)

# COV_2LR+PTP_p0 MODEL: Participant data will be fit with two group-level learning rate intercepts (alpha_OC+_INT, alpha_OC-_INT), each of which
# will be paired with one or more group-level covariate parameters (e.g., alpha_OC+_COV_PC_BPD_comm_PTSD and
# alpha_OC+_COV_PC_BPD_uniq_PTSD), as well as a single group-level p0 intercept (p0_INT), one or more group-level covariate parameters (e.g.,
# p0_COV_PC_BPD_comm_PTSD and p0_COV_PC_BPD_uniq_PTSD), and a single group-level inverse temperature (beta).

def optimize_2lr_p0_PCA_1W(params, choices, outcomes, pca_vec, smax=1):
    """
    params = [alpha_INT, alpha_COV_pc1, alpha_COV_pc2]
      - all are group-level coefficients on the *logit* scale
    pca_vec = [z_pc1, z_pc2] for THIS participant
    """
    alpha_OC_POS_INT, alpha_OC_NEG_INT, w1, p0 = params

    # the group level intercept + covariate contributions 
    x_alpha_1 = (alpha_OC_POS_INT+ (w1 * pca_vec[0])) 
    x_alpha_2 = (alpha_OC_NEG_INT+ (w1 * pca_vec[0]))
    # map to (0,1)
    lr_pos = np.exp(x_alpha_1) / (np.exp(x_alpha_1) + 1)
    lr_neg = np.exp(x_alpha_2) / (np.exp(x_alpha_2) + 1)

    p0=(np.exp(p0)/(np.exp(p0)+1))*2-1

    # generate predictions 
    mod = genModelPreds_biasOC_2LR(outcomes, lr_pos, lr_neg, p0)

    # softmax and NLL using your helpers
    actProbs = calcActProbs(mod, smax)
    nLL, _ = calcNegLogLike(choices, actProbs)

    return np.sum(nLL)

    
def neg_loglik_cov_1lr(params, trials, covs):
    """
    params = [alpha_INT, alpha_COV_comm, alpha_COV_uniq, log_beta]
    trials: dataframe-like subsettable by pid with choice/outcome columns
    covs:   dataframe-like indexed by pid with z_comm, z_uniq
    """
    alpha_INT, aC_comm, aC_uniq, log_beta = params
    beta = np.exp(log_beta)  # enforce beta > 0

    nll = 0.0

    for pid in trials["pid"].unique():
        z_comm = covs.loc[pid, "z_comm"]
        z_uniq = covs.loc[pid, "z_uniq"]

        # participant-specific learning rate implied by covariates
        x_alpha = alpha_INT + aC_comm * z_comm + aC_uniq * z_uniq
        alpha_i = sigmoid(x_alpha)

        # initialize belief
        p = 0.5

        # iterate trials for this participant in order, doing it without the prebuilt functions for my own clarity
        df_i = trials[trials["pid"] == pid].sort_values("trial_index")
        for _, row in df_i.iterrows():
            a = int(row["choice"])          # 1..5
            oc = int(row["outcome"])        # 0/1

            # compute EVs for all actions 1..5
            actions = np.arange(1, 6)
            EV = p * actions + (1 - p) * (-actions)  # = actions*(2p-1)

            # softmax over actions
            logits = EV / beta
            logits -= np.max(logits)  # numerical stability
            probs = np.exp(logits)
            probs /= probs.sum()

            # accumulate negative log-likelihood of chosen action
            p_choice = probs[a - 1]
            nll -= np.log(p_choice + 1e-12)

            # RW update
            pe = oc - p
            p = p + alpha_i * pe

    return nll


def hrarchFit_gSMbrute_iLRmin(smaxRange, chDf, ocDf, lrRange=np.linspace(-10,10,5)):
    # timer=time.time()
    smax_nll=np.inf
    # print(smaxRange)
    for smax in smaxRange:
        fit_lrDF=pd.DataFrame(index=chDf.PID.unique(), columns=chDf.cond.unique(), data=np.nan)    
        nllDF=pd.DataFrame(index=chDf.PID.unique(), columns=chDf.cond.unique(), data=np.inf)
        for PID in chDf.PID.unique():
            for cond in chDf.cond.unique():
                lr_nll = np.inf
                for lr in lrRange:

                    result=sp.optimize.minimize(optimizeLR_sft_fixedSM, (lr,),(chDf.loc[(chDf.PID==PID)&(chDf.cond==cond),'ch_01':], ocDf.loc[(ocDf.PID==PID)&(ocDf.cond==cond), 'oc_01':],smax),bounds=((-10,10),))

                    if result.fun < lr_nll:
                        lr_nll = result.fun
                        lr_val = result.x[0]
                if lr_nll < nllDF.loc[PID,cond]:
                    nllDF.loc[PID,cond]=lr_nll
                    fit_lrDF.loc[PID,cond]=np.exp(lr_val)/(np.exp(lr_val)+1)

        if nllDF.sum().sum() < smax_nll:
            smax_nll=nllDF.sum().sum()
            bestSM = smax
            bestLR = fit_lrDF.copy()

        # print('sm{} finished at: {:02f}'.format(smax,time.time()-timer))
        
    return (bestSM, bestLR)


def tiered_hrarchFit_gSMbrute_iLRmin(chDF, ocDF):
    timer=time.time()
    SML1,_ =hrarchFit_gSMbrute_iLRmin(np.linspace(0,20,6), chDF, ocDF)
    if SML1==0:
        SML2,_ =hrarchFit_gSMbrute_iLRmin(np.linspace(0,4,5), chDF, ocDF)
    else:
        SML2,_ =hrarchFit_gSMbrute_iLRmin(np.linspace(SML1-4,SML1+4,9), chDF, ocDF)

    if SML2==0:
        SML3,_ =hrarchFit_gSMbrute_iLRmin(np.linspace(0,1.2,4), chDF, ocDF)
    else:
        SML3,_ =hrarchFit_gSMbrute_iLRmin(np.linspace(SML2-1,SML2+1,6), chDF, ocDF)

    if SML3==0:
        SML4,_ =hrarchFit_gSMbrute_iLRmin(np.linspace(0,.4,5), chDF, ocDF)
    else:
        SML4,_ =hrarchFit_gSMbrute_iLRmin(np.linspace(SML3-.4,SML3+.4,9), chDF, ocDF)

    if SML4==0:
        SML5,_ =hrarchFit_gSMbrute_iLRmin(np.linspace(0,.12,4), chDF, ocDF)
    else:
        SML5,_ =hrarchFit_gSMbrute_iLRmin(np.linspace(SML4-.1,SML4+.1,6), chDF, ocDF)

    if SML5==0:
        SML6,fit_lrDF =hrarchFit_gSMbrute_iLRmin(np.linspace(0,.04,5), chDF, ocDF)
    else:
        SML6,fit_lrDF =hrarchFit_gSMbrute_iLRmin(np.linspace(SML5-.04,SML5+.04,9), chDF, ocDF)
    
    print('final sm: {} finished at: {:02f}'.format(SML6,time.time()-timer))
    
    return SML6,fit_lrDF

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

## write out all the data for upload to the cluster
# FIT THE LOTTERY and TRUST DATA TOGETHER WITH A BIASED OUTCOME (d) single LR MODEL and initialEstimate
start_time = datetime.now()

smax=1
lrRange=np.linspace(-2,2,5)
dRange=np.linspace(-2,2,5)
p0Range=np.linspace(-2,2,5)
# fit_lrDF_subj=pd.DataFrame(index=lrnEstArr.PID.unique(), columns=['subjLR', 'subjNLL', 'subjLRTp'], data=np.inf)    
# allLrnSumm=lrnSumm.add_prefix('tr_').join(rwlrnSumm.add_prefix('rw_'))
lrnEstArr=data['lrnEstArr']
lrnOCArr=data['lrnOCArr']
lrnSumm=data['lrnSumm']

subj_ids = data['subjs']  # Replace with the actual key or source for subject IDs
rwlrnEstArr=data['rwlrnEstArr']
rwlrnOCArr=data['rwlrnOCArr']
rwlrnSumm=data['rwlrnSumm']

# fit_lrDF_subj=pd.DataFrame(index=lrnEstArr.PID.unique(), columns=['subjLR', 'subjNLL', 'subjLRTp'], data=np.inf)    
#Why is this here?
allLrnSumm=lrnSumm.add_prefix('tr_').join(rwlrnSumm.add_prefix('rw_'))


fitCount=0;
parser = argparse.ArgumentParser()
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=None)
args = parser.parse_args()

pids_all = list(lrnEstArr.PID.unique())
start = max(0, args.start)
end = len(pids_all) if args.end is None else min(args.end, len(pids_all))
pids = pids_all[start:end]

print(f"SLICE: start={start}, end={end}, n={len(pids)} of total={len(pids_all)}")
for PID in pids:
    fitCount+=1
    print(f'subj_{fitCount:03.0f} ({datetime.now()-start_time}:0.2f): {PID}')
    tmpDat=lrnEstArr.query(f'PID==\"{PID}\"').join(rwlrnEstArr.query(f'PID==\"{PID}\"').drop(columns=['PID']).add_prefix('rw_'))
    allestDat=tmpDat.transpose().dropna().drop('PID').astype(int)
    allestDat.columns=['ch_{:02d}'.format(i) for i in allestDat.columns+1]
    tmpDat=lrnOCArr.query(f'PID==\"{PID}\"').join(rwlrnOCArr.query(f'PID==\"{PID}\"').drop(columns=['PID']).add_prefix('rw_'))
    allocDat=tmpDat.transpose().dropna().drop('PID').astype(int)*2-1
    all_lrs_nll=np.inf
    for lrs, ds, p0s in zip(lrRange,dRange, p0Range):
        result_s=sp.optimize.minimize(optimizeLR_sft_fixedSM_posNegOC_p0, (lrs,ds,p0s,),(allestDat, allocDat),bounds=((-10,10),(-10,10),(-1,1)))
        if result_s.fun < all_lrs_nll:
            all_lrs_nll = result_s.fun
            all_lrs_val = result_s.x[0]
            all_lrs_d = result_s.x[1]
            all_lrs_p0 = result_s.x[2]
    allLrnSumm.loc[PID,'subjLR_3prm']=np.exp(all_lrs_val)/(np.exp(all_lrs_val)+1)
    allLrnSumm.loc[PID,'subjD_3prm']=np.exp(all_lrs_d)/(np.exp(all_lrs_d)+1)
    allLrnSumm.loc[PID,'subjp0_3prm']=all_lrs_p0
    allLrnSumm.loc[PID,'subjNLL_3prm']=all_lrs_nll
    allchncNLL=np.sum(chanceLikelihood(allestDat))
    allLrnSumm.loc[PID, 'subjLRTp_3prm']=stats.chi2.sf(2*(allchncNLL-allLrnSumm.loc[PID,'subjNLL_3prm']),1)

allLrnSumm=allLrnSumm.loc[:,allLrnSumm.columns[~allLrnSumm.columns.str.startswith('qa:')].append(allLrnSumm.columns[allLrnSumm.columns.str.startswith('qa:')])]


import os

os.makedirs("results", exist_ok=True)

job_id  = os.environ.get("SLURM_JOB_ID", "local")
task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "0")

tmp_file = f"results/_tmp_{job_id}_{task_id}.csv"
final_file = "results/BiasOut1LR_ALL.csv"

# always write a temp file
allLrnSumm.loc[pids].to_csv(tmp_file)

# ONLY array task 0 assembles the final file
if task_id == "0":
    import glob
    import pandas as pd

    tmp_files = sorted(glob.glob(f"results/_tmp_{job_id}_*.csv"))
    dfs = [pd.read_csv(f, index_col=0) for f in tmp_files]
    final_df = pd.concat(dfs, axis=0)

    final_df.to_csv(final_file)
    print(f"Wrote final merged file: {final_file}")