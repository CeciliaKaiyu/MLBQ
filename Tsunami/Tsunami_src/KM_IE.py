# This script defines kernel mean and initial error for Tsunami example

import torch
import math as math

# Matern 2.5 kernel Kernel mean
def KM_Matern25(X,hyper):
    '''

    :param X:  \omega_1 \omega_2 \omega_3
    :param hyper: length-scale
    :return: kernel mean
    '''

    x1 = X[0:,0]
    x2 = X[0:,1]
    x3 = X[0:,2]

    def km_1d(x,a,b,len):

        out = (16.*torch.sqrt(torch.tensor([5.]))*len - \
         torch.exp(torch.sqrt(torch.tensor([5.]))*(a-x)/ len)*  \
         (torch.sqrt(torch.tensor([5.]))*(8.*len**2.+5.*(a-x)**2.)/len+25*(x-a))+  \
         torch.exp(torch.sqrt(torch.tensor([5.]))*(x-b)/len)*  \
         ((-torch.sqrt(torch.tensor([5.])))*  \
         (8.*len**2+5.*(x-b)**2.)/len+25.*(x-b)))/(15.*(b-a))

        return out

    KM = km_1d(x1,0.125,0.5,hyper[0]) * km_1d(x2,100.,200.,hyper[1]) * km_1d(x3,5.,15.,hyper[2])

    return KM


#Matern 2.5 kernel Initial Error
def IE_Matern25(hyper):

    '''

    :param hyper: length-scale
    :return: initial error
    '''

    def ie_1d(a,b,len):

        out = 2.*(-8.*torch.sqrt(torch.tensor([5.]))*a*len+8.*torch.sqrt(torch.tensor([5.]))*b*len-15.*len**2+ \
            torch.exp((a-b)*torch.sqrt(torch.tensor([5.]))/len)*(5.*a**2-10*a*b+5*b**2 \
         -7.*torch.sqrt(torch.tensor([5.]))*a*len+7.*torch.sqrt(torch.tensor([5.]))*b*len+15.*len**2))/(15.*(a-b)**2)

        return out

    IE = ie_1d(0.125,0.5,hyper[0]) * ie_1d(100.,200.,hyper[1]) * ie_1d(5.,15.,hyper[2])

    return IE

