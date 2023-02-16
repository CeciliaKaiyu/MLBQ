#kernel functions

import torch

#Matern 2.5 kernel for ODE example
def Matern25_ODE(X,hyper):
    '''

    :param X: \omega_1,\omega_2
    :param hyper: length-scales
    :return: Matern 2.5 covariance matrix
    '''
    #take \omega_1,\omega_2 from  X
    a=X[0:,0]
    Z=X[0:,1]
    N = a.size()[0]
    #rewrite \omega_1 as a matrix
    a=a.unsqueeze(1)
    ai=a.repeat(1,N)
    aj=torch.transpose(ai, 0, 1)
    # rewrite \omega_2 as a matrix
    Z = Z.unsqueeze(1)
    Zi=Z.repeat(1,N)
    Zj=torch.transpose(Zi, 0, 1)

    output=(1. + torch.sqrt(torch.tensor([5.])) * abs(ai - aj) / hyper[0] + 5. * (ai - aj) ** 2. / (3. * hyper[0] ** 2.)) * torch.exp(-torch.sqrt(torch.tensor([5.])) * torch.abs(ai - aj) / hyper[0]) * ( \
                1. + torch.sqrt(torch.tensor([5.])) * torch.abs(Zi - Zj) / hyper[1] + 5. * (Zi - Zj) ** 2. / (3. * hyper[1] ** 2.)) * torch.exp(
        -torch.sqrt(torch.tensor([5.])) * torch.abs(Zi - Zj) / hyper[1])

    return output




#squared exponential kernel for ODE example
def Gauss_ODE(X,hyper):
    '''

    :param X: \omega_1,\omega_2
    :param hyper: length-scales**2
    :return: SE covariance matrix
    '''
    #take \omega_1,\omega_2 from  X
    a=X[0:,0]
    Z=X[0:,1]
    N = a.size()[0]
    #rewrite \omega_1 as matrix
    a=a.unsqueeze(1)
    ai=a.repeat(1,N)
    aj=torch.transpose(ai, 0, 1)
    # rewrite \omega_2 as matrix
    Z = Z.unsqueeze(1)
    Zi=Z.repeat(1,N)
    Zj=torch.transpose(Zi, 0, 1)

    output= torch.exp(- (ai - aj) ** 2. /  hyper[0] ) * torch.exp(-(Zi - Zj)**2. / hyper[1])

    return output

