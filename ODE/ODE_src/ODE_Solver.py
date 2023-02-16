# sampler and ODE solver

import qmcpy as qmcpy
from scipy.stats import norm
from pyDOE import *
import numpy as np

def ODE_Sampler(N, method="MC", seed=7):
    '''

    :param N: sample size
    :param method: sampling method (default: IID)
    :param seed: seed used in function qmcpy.Halton
    :return: \omega_1, \omega_2
    '''

    #Step1: define a sampling strategy: QMC / MC / LHS
    if method == "Sobol":


        qmc = qmcpy.DigitalNetB2(2)

        # generate \omega_1, \omega_2
        qmc_sample = qmc.gen_samples(N)
        a = qmc_sample[0:, 0]
        z = norm.ppf(qmc_sample[0:, 1])

    elif method == "Halton":

        qmc = qmcpy.Halton(2,seed=seed,randomize='QRNG',generalize=True)

        # generate \omega_1, \omega_2
        qmc_sample = qmc.gen_samples(N)
        a = qmc_sample[0:, 0]
        z = norm.ppf(qmc_sample[0:, 1])

    elif method == "MC":

        # generate \omega_1, \omega_2
        a = np.random.uniform(0,1,N)
        z = np.random.normal(0,1,N)

    elif method == "LHS": #use LHS

        # generate \omega_1, \omega_2
        az = lhs(2, samples=N)
        a = az[0:,0]
        z = norm.ppf(az[0:,1])


    X = np.vstack((a, z)).T

    return X








def ODE_Solver(X, l):
    '''

    :param X: \omega
    :param l: level l
    :return: fine approximation, coarse approximation
    '''

    a=X[:,0]
    z=X[:,1]
    N=int(np.shape(a)[0])

    #set grid size at different level
    n = [2,4,32]

    #Step2: ODE solvers:

    Pf = np.zeros(N)  # fine approximation
    Pc = np.zeros(N)  # coarse approximation


    nf = n[l]
    hf = 1/nf


    if l>1:
        #fine
        #A0f
        cf = np.repeat(1,nf)
        A0f = np.diag(-cf[1:] - cf[:(nf-1)])
        grid = np.indices(((nf - 1), (nf - 1)))
        A0f[(grid[0] - grid[1] == 1)|(grid[0] - grid[1] == -1)]=1
        A0f=hf**(-2)*A0f

        #A1f
        cf=(np.array(range(1,(nf+1)))-0.5)*hf
        A1f=np.diag(-cf[1:] - cf[:(nf-1)])
        grid = np.indices(((nf-1),(nf-1)))
        A1f[grid[0] - grid[1] == 1]=cf[1:nf-1]
        A1f[grid[0] - grid[1] == -1]=cf[1:nf-1]
        A1f=hf**(-2)*A1f

        cf = np.repeat(1, nf-1)

        # coarse
        nc=n[l-1]
        hc=1/nc

        #A0c
        cc=np.repeat(1,nc)
        A0c=np.diag(-cc[1:] - cc[:(nc-1)])
        grid = np.indices(((nc - 1), (nc - 1)))
        A0c[(grid[0] - grid[1] == 1) | (grid[0] - grid[1] == -1)] = 1
        A0c = hc ** (-2) * A0c

        #A1c
        cc=(np.array(range(1,(nc+1)))-0.5)*hc
        A1c=np.diag(-cc[1:] - cc[:(nc-1)])
        grid = np.indices(((nc-1),(nc-1)))
        A1c[grid[0] - grid[1] == 1]=cc[1:nc-1]
        A1c[grid[0] - grid[1] == -1]=cc[1:nc-1]
        A1c=hc**(-2)*A1c

        cc = np.repeat(1, nc-1)

        for nl in range(0,N):
            U = a[nl]
            Z = z[nl]
            uf=np.linalg.inv(A0f+U*A1f)*(50*Z**2*cf)
            Pf[nl]=np.sum(hf*uf)


            uc=np.linalg.inv(A0c+U*A1c)*(50*Z**2*cc)
            Pc[nl]=np.sum(hc*uc)

    elif l==1:
        #fine
        #A0f
        cf = np.repeat(1,nf)
        A0f = np.diag(-cf[1:] - cf[:(nf-1)])
        grid = np.indices(((nf - 1), (nf - 1)))
        A0f[(grid[0] - grid[1] == 1)|(grid[0] - grid[1] == -1)]=1
        A0f=hf**(-2)*A0f

        #A1f
        cf=(np.array(range(1,(nf+1)))-0.5)*hf
        A1f=np.diag(-cf[1:] - cf[:(nf-1)])
        grid = np.indices(((nf-1),(nf-1)))
        A1f[grid[0] - grid[1] == 1]=cf[1:nf-1]
        A1f[grid[0] - grid[1] == -1]=cf[1:nf-1]
        A1f=hf**(-2)*A1f

        cf = np.repeat(1, nf-1)

        #coarse
        nc=n[l-1]
        hc=1/nc

        #A0c
        cc=np.repeat(1,nc)
        A0c=-cc[1:] - cc[:(nc-1)]
        A0c=hc**(-2)*A0c

        #A1c
        cc=(np.array(range(1,(nc+1)))-0.5)*hc
        A1c=-cc[1:] - cc[:(nc-1)]
        A1c=hc**(-2)*A1c

        cc = np.repeat(1, nc - 1)

        for nl in range(0,N):
            U = a[nl]
            Z = z[nl]
            uf = np.linalg.inv(A0f+U*A1f)*(50*Z**2*cf)
            Pf[nl] = np.sum(hf*uf)

            uc=(1/(A0c+U*A1c))*(50*Z**2*cc)
            Pc[nl]=np.sum(hc*uc)

    else:
        #fine
        #A1f
        cf = np.repeat(1, nf)
        A0f = -cf[1:] - cf[:(nf - 1)]
        A0f = hf ** (-2) * A0f

        # A1f
        cf = (np.array(range(1, (nf + 1))) - 0.5) * hf
        A1f = -cf[1:] - cf[:(nf - 1)]
        A1f = hf ** (-2) * A1f

        cf = np.repeat(1, nf - 1)

        for nl in range(0,N):
            U=a[nl]
            Z=z[nl]
            uf=(1/(A0f+U*A1f))*(50*Z**2*cf)
            Pf[nl]=np.sum(hf*uf)

            Pc[nl]=0


    return Pf,Pc
