# This script define the function that reshapes the data stored in csv file to compute the estmates of different estimators
# and the function that computes the coverage probability

import pandas as pd
import numpy as np
import scipy.stats as st

#Get Multilevel estmates
def Get_Output(data,types,N_budget=3,N_runs=100,L=3,l=3):
    '''

    :param data: data frame
    :param types: corresponding column number
    :param N_budget: the number of budget level
    :param N_runs: the number of runs
    :param L: total number of levels
    :param l: take 0-l level to compute the estimate
    :return: the multilevel etimates obtained from [budget_level_j, run_i]
    '''

    # take the column and reshape it
    datai=data[[types]]
    datai = datai.to_numpy()
    datai = np.reshape(datai, (N_runs, N_budget, L))

    # create array to save Ests
    Esti=np.zeros((N_budget, N_runs))

    for i in range(N_runs):
        for j in range(N_budget):
            Esti[j, i] = np.sum(datai[i, j, 0:l])

    return Esti



CI8=st.norm.ppf(.9)
CI6=st.norm.ppf(.8)
CI4=st.norm.ppf(.7)
CI2=st.norm.ppf(.6)


#compute coverage probability
def Coverage_Prob(Acc_Est, E,V,CI=np.array([0.2,0.4,0.6,0.8])):
    '''

    :param Acc_Est: True value
    :param E: the multilevel etimates in [budget_level_j, run_i] format
    :param V: the multilevel variance in [budget_level_j, run_i] format
    :param CI: credible level
    :return: covarage probability
    '''

    # the number of budget levels, change of L, and number of runs
    n_level=E.shape[0]


    # the corresponding Z score and the number of scores
    CI_score=st.norm.ppf(1.-(1.-CI)/2)
    nCI=CI.shape[0]

    # create an array to store result
    Cov_Prob=np.zeros((n_level,nCI))

    if np.sum(V<0)==0:

        # Iterate for different budget sizes

        for i in range(n_level):

            # compute coverage probabilty at credible level CI[nCI]

            for k in range(nCI):
                Zscore = CI_score[k]
                TFArray = np.logical_and(Acc_Est > (E[i, :] - Zscore * np.sqrt(V[i, :])),
                                         Acc_Est < (E[i, :] + Zscore * np.sqrt(V[i, :])))
                Cov_Prob[i, k] = np.mean(TFArray)

    else:
        Cov_Prob=np.zeros((n_level,nCI))-1.

    return Cov_Prob
