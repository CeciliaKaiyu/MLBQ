# This script defines kernel functions that we can use in Tsunami example

import torch


# Matern 2.5 kernel
def Matern25(X,hyper):
    '''

    :param X:  \omega_1 \omega_2 \omega_3
    :param hyper: length-scale
    :return: Matern 2.5 covariance matrix
    '''

    # take \omega_1 \omega_2 \omega_3 from input tensor X
    x1 = X[0:,0]
    x2 = X[0:,1]
    x3 = X[0:,2]

    def Matern25_1d(x,len):
        # rewrite x as matrix
        N = x.size()[0]
        x = x.unsqueeze(1)
        xi = x.repeat(1,N)
        xj = torch.transpose(xi, 0, 1)

        out = (1. + torch.sqrt(torch.tensor([5.])) * abs(xi - xj) / len + 5. * (xi - xj) ** 2. / (3. * len ** 2.)) * torch.exp(-torch.sqrt(torch.tensor([5.])) * torch.abs(xi - xj) / len)

        return out

    output = Matern25_1d(x1,hyper[0]) * Matern25_1d(x2,hyper[1]) * Matern25_1d(x3,hyper[2])

    return output
