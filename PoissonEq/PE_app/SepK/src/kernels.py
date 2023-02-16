import torch
from torch import nn


def sep_kernel(X,hyper):
    '''

    :param X: list of x
    :param Y: list of y
    :param hyper: a tensor (L+1)*(L+2) [:L+1,:L+1] is B, [:L+1,L+1] is repeated hyper
    :param inner_k(x,hyper)
    :return:
    '''

    # compute levels
    L = len(X)
    N_list = [X[0].size()[0]]
    for l in range(1,L):
        N_list.append(X[l].size()[0])

    #get b and hyper

    b = torch.ones((L,L))*(0.05)
    for i in range(L):
            b[i,i]=1.


    # def inner_kernel
    def inner_k(X, hyper):
        '''

        :param X: input
        :param hyper:  length-scale
        :return: Matern 0.5 covariance matrix
        '''

        x = X
        len = hyper

        # rewrite x as matrix
        N = x.size()[0]
        xi = x.repeat(1, N)
        xj = torch.transpose(xi, 0, 1)

        out = torch.exp(- torch.abs(xi - xj) / len)

        return out

    X_tensor=X[0]
    for i in range(1,L):
        X_tensor=torch.cat((X_tensor, X[i]), 0)

    B=torch.zeros((sum(N_list),sum(N_list)))

    N0_list = [0]
    N0_list = N0_list + N_list
    BN_list = []
    for i in range(1, L + 2):
        BN_list = BN_list + [sum(N0_list[0:i])]

    for i in range(L):
        for j in range(L):
            B[BN_list[i]:BN_list[i + 1], BN_list[j]:BN_list[j + 1]] = b[i, j]

    BK= inner_k(X=X_tensor, hyper=hyper) * B

    return BK






