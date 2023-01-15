# MLBQ estimator for ODE example [ when the kernel mean has a closed form but the initial error does not have a closed form ]
from ODE.ODE_src.BQsp import *
import numpy as np


def MLBQsp(L, N, solver, sampler, X2, hyper, kernel, KM, IE1, KM2, CalMLMC=True,  sum=True, nuggetL=None, method="LBFGS", epochL=None, bs=None, maxBatch_size=5, lrL=None, betasL=None, epsL=None, weight_decayL=None, max_iterL=None):
    '''

    :param L: number of levels
    :param N: sample size in each level
    :param solver: numerical solver return fine and coarse output Y in np array
    :param sampler: sample input variable X in np array
    :param X2: input variable whose initial errors are not available in closed form
    :param hyper: initial hyper parameter value for optimization
    :param kernel: self-specified kernel function, such as Matern25, SE
    :param KM: closed form kernel mean function of X (all input variable)
    :param IE1: initial error of variables whose initial errors have closed form
    :param KM2: kernel mean of X2 (closed form)
    :param CalMLMC: Calculate MLMC estimate when it is True (default: True)
    :param sum: return MLBQ estimate when sum=True and estimate at each level when sum=False (default: True)
    :param nuggetL: a vector of nugget in each level, if it is None, nugget is 1e-6 in all levels
    :param method: optimizer used to tune hyper parameters, can be "Adam" or "LBFGS" (default: "LBFGS" )
    :param epochL: Number of epochs in each level used in training loop to tune hyper-parameters (if set to None use default 20 in all levels)
    :param bs: a vector of specified batch size for each level , if it is none, set bs to be the min of (sample_size, minBatch_size)
    :param maxBatch_size: maxmum batch size (default:5)
    :param lrL: a vector of learning rate (if set to None use default 0.01 in all levels)
    :param betasL: (only used for "Adam") a array of coefficients used for computing running averages of gradient and its square (if set to None use default (0.9, 0.999) in all levels)
    :param epsL: (only used for "Adam") term added to the denominator to improve numerical stability (if set to None use default 1e-08 in all levels)
    :param weight_decayL: (only used for "Adam") weight decay (if set to None use default 0 in all levels)
    :param max_iterL: (only used for "LBFGS") a vector of maximal number of iterations per optimization step (if set to None use default 20 in all levels)

    :return:

        Est[l,0]:Mean (BQ estimate) at level l
        Est[l,1]:Variance at level l

        Est[l,2]:MC Mean at level l
        Est[l,3]:MC Variance at level l

   '''

    # check if compute MC estimate
    if CalMLMC:
        Est = np.zeros((L, 4))
    else:
        Est = np.zeros((L, 2))

    # specify nugget
    if nuggetL is None:
        nuggetL = [1e-6]*L

    # specify max_iterL
    if max_iterL is None:
        max_iterL = [20]*L

    # specify epoch
    if epochL is None:
        epochL = [20]*L

    # specify batch size
    if bs is None:
        bs = np.zeros(L)
        for l in range(L):
            bs[l] = int(torch.min(torch.tensor([N[l], maxBatch_size])))

    # specify learning rate
    if lrL is None:
        lrL = [0.01]*L

    # specify coefficients
    if betasL is None:
        betasL = [(0.9, 0.999)]*L

    # specify eps
    if epsL is None:
        epsL = [1e-08]*L

    # specify eps weight decay
    if weight_decayL is None:
        weight_decayL = [0]*L

    # iterate over levels
    for l in range(L):
        # obtain input and output variables
        X = sampler(N[l])
        Yf,Yc = solver(X,l)

        # convert np array to tensor
        Y = torch.from_numpy(Yf-Yc)
        X = torch.from_numpy(X)

        #obtain BQ estimate at level l
        E, V = BQsp(X=X, X2=X2, Y=Y, Gram=Gram, hyper=hyper, kernel=kernel, KM=KM, IE1=IE1, KM2=KM2,  nugget=nuggetL[l], method=method, epoch=epochL[l], batch_size=int(bs[l]),  lr=lrL[l], betas=betasL[l], eps=epsL[l], weight_decay=weight_decayL[l], max_iter = max_iterL[l])
        Est[l,0] = float(E)
        Est[l,1] = float(V)

        if CalMLMC:
            # obtain MC estimate at level l
            Est[l,2] = float(torch.mean(Y))
            Est[l,3]= float(torch.var(Y)/N[l])

    # if sum is True, return MLBQ estimate else return BQ estimate at each level
    if sum:
        return Est.sum(axis=0)
    else:
        return Est


