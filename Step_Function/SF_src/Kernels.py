#kernel functions
import torch
import math

# Matern 0.5 kernel
def Matern05(X,hyper):
    '''

    :param X:  \omega
    :param hyper: length-scale
    :return: Matern 0.5 covariance matrix
    '''

    N = X.size()[0]
    X = X.unsqueeze(1)
    Xi = X.repeat(1, N)
    Xj = torch.transpose(Xi, 0, 1)

    output = torch.exp(- torch.abs(Xi - Xj) / hyper)

    return output


# SE kernel
def Gauss(X,hyper):
    '''

    :param X:  \omega
    :param hyper: length-scale**2
    :return: SE covariance matrix
    '''

    # rewrite x as matrix
    N = X.size()[0]
    X = X.unsqueeze(1)
    Xi=X.repeat(1,N)
    Xj=torch.transpose(Xi, 0, 1)

    output = torch.exp(- (Xi - Xj) ** 2. /  hyper )

    return output






