# This script define all kernel functions that we will need to use in this Poisson equation example

import torch

# Matern 0.5 kernel for Poisson equation example
def Matern05(X,hyper):
    '''

    :param X: input variable
    :param hyper:  length-scale
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
