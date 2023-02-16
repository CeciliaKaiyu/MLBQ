#This script solve coefficient vector alpha of length p_l from the linear system
import numpy as np


def coef_vec(h,a):
    '''

    :param h: grid size of xi (p must be int, which is the number of xi)
    :param a: input variable
    :return:
     u: finite element approximation
     alpha: coefficient vector
    '''

    # number of xi given the grid size h

    p = 1/h

    if p%1>0:
        raise ValueError("Need 1/h to be integer")

    p = int(p)

    #construct L

    L = np.eye(p-1)*(2/h)

    gridL = np.indices(((p-1),(p-1)))

    L[gridL[0] - gridL[1] == -1] = np.ones(p-2)*(-1/h)
    L[gridL[0] - gridL[1] == 1] = np.ones(p-2)*(-1/h)

    #compute g and vi

    g = np.ones((p-1))*h
    vi = np.zeros((p-1))

    for i in range(1,p):
        v1 = a - (i-1) * h
        v2 = i * h - a
        v3 = -v2
        v4 = (i+1) * h - a
        if v1>=0 and v2>0:
            vi[i-1]= v1 / h
        if v3 >= 0 and v4 > 0:
            vi[i - 1] = v4 / h

    alpha = np.linalg.inv(-L)@g

    #compute approximation of ul
    u = np.sum(alpha*vi)

    return(u,alpha)
