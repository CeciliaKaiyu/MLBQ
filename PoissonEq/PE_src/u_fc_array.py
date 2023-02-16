# Function u_fc computes fine and coarse approximation
# h is 1-d at level 0 and 2-d at levels > 0
from PoissonEq.PE_src.Def_alpha import *

def u_fc(h,a):
    '''
    :param h: grid size of xi (p must be int, which is the number of xi)
    :param a: input variable
    :return: fine and coarse approximation
    '''

    # sample size
    n = np.shape(a)[0]
    # specify uc and uf array
    uc = np.zeros(n)
    uf = np.zeros(n)

    # check if level is level 0

    if h[0] == 1:
        # if it is level 0, only compute uf and let uc = 0
        for i in range(n):
            uf[i],  alpha = coef_vec(h=h[1], a=a[i])

    else:
        # compute uf and uc with the sample input
        for i in range(n):
            uf[i],  alpha = coef_vec(h=h[1], a=a[i])
            uc[i],  alpha = coef_vec(h=h[0], a=a[i])

    return(uf,uc)
