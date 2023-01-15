# BQ estimator for ODE example [ when the kernel mean has a closed form but the initial error does not have a closed form ]

from src.Optimizer import *

def BQsp(X, X2, Y, Gram, hyper, kernel, KM, IE1, KM2, nugget=1e-6, method="LBFGS", epoch=20, batch_size=5, lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, max_iter=20):
    '''

    :param X: all input variable (should be tensor)
    :param X2: input variable whose initial errors are not available in closed form
    :param Y: output variable(should be tensor)
    :param Gram: the gram matrix function
    :param hyper: initial hyper parameter value for optimization
    :param kernel: self-specified kernel function, such as Matern25, SE
    :param KM: closed form kernel mean function of X (all input variable)
    :param IE1: initial error of variables whose initial errors have closed form
    :param KM2: kernel mean of X2 (closed form)
    :param nugget: nugget (default: 1e-6)
    :param method: optimizer used to tune hyper parameters, can be "Adam" or "LBFGS" (default: "LBFGS" )
    :param epoch: (only used for "Adam") Number of epochs used in training loop to tune hyper-parameters (default: 20)
    :param batch_size: (only used for "Adam") batch size used in training loop to tune hyper-parameters (default: 5)
    :param lr: learning rate (default: 0.01)
    :param betas: (only used for "Adam") coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999))
    :param eps: (only used for "Adam") term added to the denominator to improve numerical stability (default: 1e-08)
    :param weight_decay: (only used for "Adam") weight decay (default: 0)
    :param max_iter: (only used for "LBFGS") maximal number of iterations per optimization step (default: 20)

    :return: E:Mean (BQ estimate) V:Variance

    '''

    # Step1: Given X, Y at level l, compute hyper-parameters

    # use training loop to tune hyper-parameters
    Hyper_cp = training_loop(X=X, Y=Y, Gram=Gram, hyper=hyper, kernel=kernel, method=method, epoch=epoch, batch_size=batch_size,lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, max_iter=max_iter)
    Hyper = Hyper_cp.detach().clone()

    # Compute the amplitude with the closed form expression
    K = kernel(X=X, hyper=Hyper) + nugget*torch.eye(X.size(0))
    Inv_K = torch.inverse(K)
    amp=Y@Inv_K@Y/Y.size(0)

    # update kernel matrix by multiplying the amplitude
    K=amp*(K-nugget*torch.eye(X.size(0))) + nugget*torch.eye(X.size(0))
    Inv_K = torch.inverse(K)

    # Step2: compute the closed form solution of BQ

    # Compute the kernel mean and the initial error with optimized hyper parameters
    km = amp*KM(X=X, hyper=Hyper)

    # Compute BQ estimate and the variance
    E = km @ Inv_K @ Y

    ie1 = IE1(hyper=Hyper[0])
    ie2 = torch.mean(KM2(X2,Hyper[1]))

    ie = amp*ie1*ie2
    V = ie - km @ Inv_K @ km
    return E, V



