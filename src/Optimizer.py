#optimizer used to tune hyper-parameters of BQ for level l

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
        # make hyper torch parameters
        self.hyper= nn.Parameter(hyper,requires_grad=True)

    def forward(self, X, kernel):
        '''

        :param X: input variable
        :param kernel: a kernel function (should be defined and provided e.g. Matern)
        :return: covariance matrix
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
    #compute the amplitude with the closed form expression
    amp=Y@Inv_K@Y/Y.size(0)
    #update kernel matrix by multiplying the amplitute
    K=amp*K
    # compute the inverse matrix
    Inv_K = Inv_K/amp
    #compute the -loglikelihood
    output=Y@Inv_K@Y+torch.logdet(K)
    return output



# Define the training loop
def training_loop(X, Y, Gram, hyper, kernel, method="LBFGS", epoch=20, batch_size=5, lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, max_iter=20):
    '''

    :param X: input variable (should be tensor)
    :param Y: output variable(should be tensor)
    :param Gram: the gram matrix function
    :param hyper: initial hyper parameter value for optimization
    :param kernel: self-specified kernel function, such as Matern25, SE
    :param method: can be "Adam" or "LBFGS" (default: "LBFGS")
    :param epoch: (only used for "Adam") Number of epochs used in training loop to tune hyer-parameters (default: 20)
    :param batch_size: (only used for "Adam") batch size used in training loop to tune hyer-parameters (default: 5)
    :param lr: learning rate (default: 0.01)
    :param betas: (only used for "Adam") coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999))
    :param eps: (only used for "Adam") term added to the denominator to improve numerical stability (default: 1e-08)
    :param weight_decay: (only used for "Adam") weight decay (default: 0)
    :param max_iter: (only used for "LBFGS") maximal number of iterations per optimization step (default: 20)
    :return: optimized hyper parameter value (except amplitude)
    '''

    #optimization method 1: LBFGS

    if method== "LBFGS":

        K = Gram(hyper)

        optimizer = torch.optim.LBFGS(K.parameters(), lr=lr, max_iter=max_iter)

        def closure():

            GramX = K(X, kernel)
            loss = NegllkLoss(GramX, Y)
            optimizer.zero_grad()
            loss.backward()

            return loss

        optimizer.step(closure)

    # optimization method 2: ADAM

    if method == "Adam":

        K = Gram(hyper)

        # number of samples
        N = X.size()[0]
        # number of batches
        nbatch = (N - 1) // batch_size + 1

        optimizer = torch.optim.Adam(K.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        # iterate over epoches
        for i in range(epoch):
            # shuffle data indices
            indices = torch.randperm(N)
            # shuffle X and Y
            RandX = X[indices,]
            RandY = Y[indices]
            # iterate over batches
            print(K.hyper)
            for j in range(nbatch):
                # split data to batches
                start_j = j * batch_size
                end_j = start_j + batch_size
                Xbatch = RandX[start_j:end_j, ]
                Ybatch = RandY[start_j:end_j]
                # use batch data to update parameters
                GramX = K(Xbatch, kernel)
                loss = NegllkLoss(GramX, Ybatch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    return K.hyper








