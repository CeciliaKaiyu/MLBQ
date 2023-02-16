#Kernel Mean and Initial Error

import torch
import math

# SE Kernel mean
def KM_Gauss(X,hyper):
    '''

    :param X:  \omega
    :param hyper: length-scale**2
    :return: kernel mean
    '''

    a=0
    b=10.

    h1 = torch.sqrt(hyper)
    out = torch.sqrt(torch.tensor(math.pi)) * h1 * (torch.erf((X-a)/h1)+torch.erf((b-X)/h1))/(2*(b-a))
    return out





#SE kernel Initial Error
def IE_Gauss(hyper):
    '''

    :param hyper: length-scale**2
    :return: initial error
    '''
    a=0
    b=10.

    h1 = torch.sqrt(hyper)
    out = h1*((-1+torch.exp(-(a-b)**2./hyper))*h1+(a-b)*torch.sqrt(torch.tensor(math.pi))*torch.erf((a-b)/h1))/(b-a)**2.

    return out




# Matern 0.5 kernel Kernel mean
def KM_Matern05(X,hyper):
    '''

    :param X: \omega
    :param hyper: length-scale
    :return: kernel mean
    '''

    a=0
    b=10.

    out =( 2.*hyper-torch.exp((a-X)/hyper)*hyper-torch.exp((X-b)/hyper)*hyper )/(b-a)

    return out





# Matern 0.5 kernel Initial Error
def IE_Matern05(hyper):
    '''

    :param hyper: length-scale
    :return: initial error
    '''
    a=0
    b=10.

    out = 2*hyper*(b-a-hyper+torch.exp((a-b)/hyper)*hyper) / (b-a)**2

    return out

