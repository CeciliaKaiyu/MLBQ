# kernel mean and initial error for Poisson equation example
# Matern 0.5

import torch


# Kernel Mean 
def KM_Matern05(X,hyper):
    '''

    :param X: input variable
    :param hyper: length-scale
    :return: kernel mean
    '''

    a=0
    b=1

    out =( 2.*hyper-torch.exp((a-X)/hyper)*hyper-torch.exp((X-b)/hyper)*hyper )/(b-a)

    return out




#Initial Error
def IE_Matern05(hyper):
    '''

    :param hyper:  length-scale
    :return: inital error
    '''
    a=0
    b=1

    out = 2*hyper*(b-a-hyper+torch.exp((a-b)/hyper)*hyper) / (b-a)**2

    return out
