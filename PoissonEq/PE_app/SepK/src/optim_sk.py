#optimizer

import torch
from torch import nn


#Define the Gram Matrix for gradient optimization
class Gram(nn.Module):
    '''
    custom Gram Matrix for gradient optimization
    '''
    def __init__(self,hyper):
        super().__init__()
        # initialize hyperparameters with hyper
        self.hyper= nn.Parameter(hyper,requires_grad=True)

    def forward(self, X, kernel):
        '''

        :param X: input variable
        :param kernel: a kernel function (should be defined and provided e.g. Matern)
        :return: covairance matrix
        '''
        K = kernel(X, self.hyper)

        return K


#Define the loss function to be -loglikelihood itself
def NegllkLoss(K,Y):
    '''

    :param K: Gram matrix
    :param Y: output variable
    :return: negative loglikelihood
    '''

    #compute the inverse matrix
    Inv_K = torch.inverse(K)
    Y_tensor = Y[0]
    L = len(Y)
    for i in range(1,L):
        Y_tensor=torch.cat((Y_tensor, Y[i]), 0)
    Y_tensor=Y_tensor.squeeze()
    #compute the amplitude with the closed form expression
    amp=Y_tensor@Inv_K@Y_tensor/Y_tensor.size(0)
    #update kernel matrix by multiplying the amplitute
    K=amp*K
    # compute the inverse matrix
    Inv_K = Inv_K/amp
    #compute the -loglikelihood
    output=Y_tensor@Inv_K@Y_tensor+torch.logdet(K)
    return output



# Define the training loop
def training_loop(X, Y, Gram, hyper, kernel,  lr=0.01, max_iter=20):
    '''

    :param X: input variable (should be tensor)
    :param Y: output variable(should be tensor)
    :param Gram: the gram matrix function
    :param hyper: initial hyper parameter value for optimization
    :param kernel: self-specified kernel function, such as Matern25, Gauss
    :param lr: learning rate (default: 0.01)
    :param max_iter: (only used for "LBFGS") maximal number of iterations per optimization step (default: 20)
    :return: optimized hyper parameter value (except amplitude)
    '''

    #optimization LBFGS

    K = Gram(hyper)

    optimizer = torch.optim.LBFGS(K.parameters(), lr=lr, max_iter=max_iter)

    def closure():

        GramX = K(X, kernel)
        loss = NegllkLoss(GramX, Y)
        optimizer.zero_grad()
        loss.backward()

        return loss

    optimizer.step(closure)


    return K.hyper








