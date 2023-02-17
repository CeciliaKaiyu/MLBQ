import torch
import math as math
import numpy as np


# Matern 0.5 kernel for Poisson equation example
def Matern05(X,hyper):
    '''

    :param X: input variable
    :param hyper: hyper parameters except amplitude: length-scale
    :return: Matern 0.5 covariance matrix
    '''
    x=X
    len=hyper

    # rewrite x as matrix
    N = x.size()[0]
    x = x.unsqueeze(1)
    xi = x.repeat(1,N)
    xj = torch.transpose(xi, 0, 1)

    out = torch.exp(- torch.abs(xi - xj) / len)

    return out

# Matern 0.5 kernel for GP
def k_matern(xi,xj,hyper):
    Ni = xi.size()[0]
    Nj = xj.size()[0]

    # rewrite X as matrix
    Xi = xi.unsqueeze(1)
    Xi = Xi.repeat(1,Nj)

    Xj = xj.unsqueeze(1)
    Xj = Xj.repeat(1,Ni)
    Xj = torch.transpose(Xj, 0, 1)

    output= torch.exp(- torch.abs(Xi - Xj)  /  hyper )

    return output

#GP posterior mean
def gp_mean(x,X,hyper,y):

    output = k_matern(x,X,hyper)@torch.inverse(k_matern(X,X,hyper))@y
    return output

#GP posterior covariance
def gp_cov(x,X,hyper):

    output = k_matern(x,x,hyper)-k_matern(x,X,hyper)@torch.inverse(k_matern(X,X,hyper))@k_matern(X,x,hyper)
    return output


# Kernel Mean
def km(X,hyper):
    '''

    :param X: input variable
    :param hyper: length-scale
    :return: kernel mean
    '''
    x=X
    len=hyper
    a=0
    b=1

    out =( 2.*len-torch.exp((a-x)/len)*len-torch.exp((x-b)/len)*len )/(b-a)

    return out

#Initial Error
def ie(hyper):
    '''

    :param hyper:  length-scale
    :return: inital error
    '''

    len=hyper

    a=0
    b=1

    out = 2*len*(b-a-len+torch.exp((a-b)/len)*len) / (b-a)**2

    return out
