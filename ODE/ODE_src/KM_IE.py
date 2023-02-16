#  Kernel mean and initial error for ODE example

import torch
import math as math


#Kernel mean when using Matern 2.5 kernel for ODE example with different length-scale
def KM_Matern25_ODE(X,hyper):
    '''

    :param X:  \omega_1 and \omega_2
    :param hyper: length-scale
    :return: kernel mean
    '''
    #take \omega_1 and \omega_2 from  X
    a=X[0:,0]
    Z=X[0:,1]

    #take hyperparameters

    h1=hyper[0] #length-scale of \omega_1 \gamma_1
    h2=hyper[1] #length-scale of \omega_2 \gamma_2

    #KM for \omega_1
    KMa=(16.*torch.sqrt(torch.tensor([5.]))*h1 - \
         torch.exp(-torch.sqrt(torch.tensor([5.]))*a/ h1)*  \
         (torch.sqrt(torch.tensor([5.]))*(8.*h1**2.+5.*a**2.)/h1+25*a)+  \
         torch.exp(torch.sqrt(torch.tensor([5.]))*(a-1.)/h1)*  \
         ((-torch.sqrt(torch.tensor([5.])))*  \
         (8.*h1**2+5.*(a-1.)**2.)/h1+25.*(a-1.)))/15.

    #KM for \omega_2
    KMZ= torch.exp(-Z**2./2.) * (  \
        4.*torch.sqrt(torch.tensor([5.]))*h2*(-5.+3.*h2**2.) +  \
        torch.sqrt(2.*torch.tensor(math.pi))*(  \
        torch.exp((torch.sqrt(torch.tensor([5.]))-h2*Z)**2./(2*h2**2.)) *  \
        (25.+3.*h2**4.-10.*torch.sqrt(torch.tensor([5.]))*h2*Z+  \
         3.*torch.sqrt(torch.tensor([5.]))*h2**3.*Z+5.*h2**2.*(-2.+Z**2.)) *  \
        torch.erfc((torch.sqrt(torch.tensor([5.]))/h2-Z)/torch.sqrt(torch.tensor([2.]))) +  \
        torch.exp((torch.sqrt(torch.tensor([5.]))+h2*Z)**2./(2*h2**2.))*  \
        (25.+3.*h2**4.+10.*torch.sqrt(torch.tensor([5.]))*h2*Z-  \
         3*torch.sqrt(torch.tensor([5.]))*h2**3.*Z+5.*h2**2.*(-2.+Z**2.))*  \
        torch.erfc((torch.sqrt(torch.tensor([5.]))/h2+Z)/  \
        torch.sqrt(torch.tensor([2.])))))/(6.*h2**4.*torch.sqrt(2.*torch.tensor(math.pi)))

    KM=KMa*KMZ

    return KM

#kernel mean of \omega_2 
def KM_Matern25_ODE2(X,hyper):
    '''

    :param X: \omega_2
    :param hyper: length-scale
    :return: kernel mean
    '''

    #take \omega_2
    Z=X

    #take the hyperparameter

    h2=hyper #length-scale of \omega_2 \gamma_2

    #KM for \omega_2
    KMZ= torch.exp(-Z**2./2.) * (  \
        4.*torch.sqrt(torch.tensor([5.]))*h2*(-5.+3.*h2**2.) +  \
        torch.sqrt(2.*torch.tensor(math.pi))*(  \
        torch.exp((torch.sqrt(torch.tensor([5.]))-h2*Z)**2./(2*h2**2.)) *  \
        (25.+3.*h2**4.-10.*torch.sqrt(torch.tensor([5.]))*h2*Z+  \
         3.*torch.sqrt(torch.tensor([5.]))*h2**3.*Z+5.*h2**2.*(-2.+Z**2.)) *  \
        torch.erfc((torch.sqrt(torch.tensor([5.]))/h2-Z)/torch.sqrt(torch.tensor([2.]))) +  \
        torch.exp((torch.sqrt(torch.tensor([5.]))+h2*Z)**2./(2*h2**2.))*  \
        (25.+3.*h2**4.+10.*torch.sqrt(torch.tensor([5.]))*h2*Z-  \
         3*torch.sqrt(torch.tensor([5.]))*h2**3.*Z+5.*h2**2.*(-2.+Z**2.))*  \
        torch.erfc((torch.sqrt(torch.tensor([5.]))/h2+Z)/  \
        torch.sqrt(torch.tensor([2.])))))/(6.*h2**4.*torch.sqrt(2.*torch.tensor(math.pi)))

    return KMZ



#intial error of \omega_1 
def IE_Matern25_ODE1(hyper):
    '''

    :param hyper: length-scale
    :return: initial error
    '''
    IEa=2.*(8.*torch.sqrt(torch.tensor([5.]))*hyper-15.*hyper**2+torch.exp(-torch.sqrt(torch.tensor([5.]))/hyper)*(5.+7.*torch.sqrt(torch.tensor([5.]))*hyper+15.*hyper**2))/15.
    return IEa



#Kernel mean when using Gaussian kernel  
def KM_Gauss_ODE(X,hyper):
    '''

    :param X:  \omega_1 and \omega_2
    :param hyper: length-scale**2
    :return: kernel mean
    '''
    #take \omega_1 and \omega_2 from input tensor X
    a=X[0:,0]
    Z=X[0:,1]

    h1=torch.sqrt(hyper[0]) #lengthscale of \omega_1 \gamma_1
    h2=torch.sqrt(hyper[1]) #lengthscale of \omega_1 \gamma_2

    KM =  h1 * (torch.erf((1.-a)/h1)+torch.erf(a/h1)) * \
            h2 * torch.exp(-Z ** 2. / (hyper[1] + 2.)) / torch.sqrt(hyper[1] + 2.) / 2. * torch.sqrt(torch.tensor(math.pi))

    return KM


#Initial Error when using Gaussian kernel
def IE_Gauss_ODE(hyper):
    '''

    :param hyper: length-scale**2
    :return: initial error
    '''

    h1=torch.sqrt(hyper[0]) #lengthscale of \omega_1 \gamma_1
    h2=torch.sqrt(hyper[1]) #lengthscale of \omega_2 \gamma_2

    IE = (hyper[0]*torch.exp(-1./hyper[0])-h1*(h1-torch.sqrt(torch.tensor(math.pi))* torch.erf(1./h1))) * \
         h2  / torch.sqrt( (hyper[1] + 4.))

    return IE




