# MLBQ estimator separable kernels

from PoissonEq.PE_app.SepK.src.optim_sk import *

def BQ_sk(X, Y, Gram, hyper, kernel, KM, CalIE=False, IE=None, nugget=1e-6, lr=0.01,  max_iter=20):
    '''

    :param X: list of x
    :param Y: list of y
    :param Gram: the gram matrix function
    :param hyper: initial hyper parameter value for optimization
    :param kernel: self-specified kernel function, such as Matern25, Gauss
    :param KM: kernel mean function (closed form)
    :param CalIE: calculate initial error and return variance when CalIE is True, otherwise False (default: False)
    :param IE: If CalIE is True, then compute initial error (closed form)
    :param lr: learning rate (default: 0.01)
    :param max_iter: (only used for "LBFGS") maximal number of iterations per optimization step (default: 20)

    :return: E:Mean (BQ estimate) V:Variance

    '''

    #L+1
    L = len(X)

    #X and Y tensor
    X_tensor=X[0]
    for i in range(1,L):
        X_tensor=torch.cat((X_tensor, X[i]), 0)

    Y_tensor = Y[0]
    for i in range(1,L):
        Y_tensor=torch.cat((Y_tensor, Y[i]), 0)
    Y_tensor=Y_tensor.squeeze()

    # Step1: Given X, Y at level l, compute hyper-parameters

    # use training loop to tune hyper-parameters
    Hyper_cp = training_loop(X=X, Y=Y, Gram=Gram, hyper=hyper, kernel=kernel, lr=lr, max_iter=max_iter)
    Hyper = Hyper_cp.detach().clone()

    b = torch.ones((L,L))*(0.05)
    for i in range(L):
            b[i,i]=1.

    N_list = []
    for l in range(L):
        N_list.append(X[l].size()[0])

    B = torch.zeros((sum(N_list), sum(N_list)))

    # Compute the amplitude with the closed form expression
    K = kernel(X=X, hyper=Hyper) + nugget*torch.eye(sum(N_list))
    Inv_K = torch.inverse(K)
    amp=Y_tensor@Inv_K@Y_tensor/Y_tensor.size(0)

    # Update kernel matrix by multiplying the amplitude
    K=amp*(K-nugget*torch.eye(sum(N_list))) + nugget*torch.eye(sum(N_list))
    Inv_K = torch.inverse(K)

    # Step2: compute the closed form solution of BQ

    N0_list = [0]
    N0_list = N0_list + N_list
    BN_list = []
    for i in range(1, L + 2):
        BN_list = BN_list + [sum(N0_list[0:i])]

    for i in range(L):
        for j in range(L):
            B[BN_list[i]:BN_list[i + 1], BN_list[j]:BN_list[j + 1]] = b[i, j]

    BlL = torch.zeros((sum(N_list), 1))
    blL = torch.sum(b, 1)
    for i in range(L):
        BlL[BN_list[i]:BN_list[i + 1]] = blL[i]

    # Compute the kernel mean and the initial error with optimized hyper parameters
    km = amp*KM(X=X_tensor, hyper=Hyper)
    km = BlL * km
    km = km.squeeze(1)

    # Compute BQ estimate
    E = km @ Inv_K @ Y_tensor

    # If CalIE is True, then compute the initial error and the variance
    if CalIE:
        ie = amp*IE(hyper=Hyper)
        ie= torch.sum(blL)*ie
        V = ie - km @ Inv_K @ km
        return E, V
    else:
        return E


