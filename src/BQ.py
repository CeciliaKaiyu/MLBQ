# BQ estimator

from src.Optimizer import *

def BQ(X, Y, Gram, hyper, kernel, KM, CalIE=False, IE=None, nugget=1e-6, method="LBFGS", epoch=20, batch_size=5, lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, max_iter=20):
    '''

    :param X: input variable (should be tensor)
    :param Y: output variable(should be tensor)
    :param Gram: the gram matrix function
    :param hyper: initial hyper parameter value for optimization
    :param kernel: self-specified kernel function, such as Matern25, SE
    :param KM: kernel mean function (closed form)
    :param CalIE: calculate initial error and return variance when CalIE is True, otherwise False (default: False)
    :param IE: If CalIE is True, then compute initial error (closed form)
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
    Hyper_cp = training_loop(X=X, Y=Y, Gram=Gram, hyper=hyper, kernel=kernel, method=method, epoch=epoch, batch_size=batch_size, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, max_iter=max_iter)
    Hyper = Hyper_cp.detach().clone()

    # Compute the amplitude with the closed form expression
    K = kernel(X=X, hyper=Hyper) + nugget*torch.eye(X.size(0))
    Inv_K = torch.inverse(K)
    amp=Y@Inv_K@Y/Y.size(0)

    # Update kernel matrix by multiplying the amplitude
    K=amp*(K-nugget*torch.eye(X.size(0))) + nugget*torch.eye(X.size(0))
    Inv_K = torch.inverse(K)

    # Step2: compute the closed form solution of BQ

    # Compute the kernel mean and the initial error with optimized hyper parameters
    km = amp*KM(X=X, hyper=Hyper)

    # Compute BQ estimate
    E = km @ Inv_K @ Y

    # If CalIE is True, then compute the initial error and the variance
    if CalIE:
        ie = amp*IE(hyper=Hyper)
        V = ie - km @ Inv_K @ km
        return E, V
    else:
        return E


