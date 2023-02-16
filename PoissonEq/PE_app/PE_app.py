# This script implements MLBQ

from src.MLBQ import *
from PoissonEq.PE_src.Kernels import *
from PoissonEq.PE_src.KM_IE import *
from PoissonEq.PE_src.u_fc_array import *

import csv


# create the solver
def PE_solver(X,l):
    hl = np.array([0.5,0.2,0.05])
    if l > 0:
        print(l)
        h = hl[l - 1:l + 1]
        uf, uc = u_fc(h, X)


    else:
        print(l)
        h = [1, hl[0]]
        uf, uc = u_fc(h, X)

    return uf, uc

# create the sampler
def PE_S3S4sampler(N):
    X = np.random.uniform(0, 1, int(N))
    return X
# can change the sampler to use regular grid

#define sample size to be n^MLMC
MLN=np.array(([67,11,1],[133,23,2],[166,46,3]))
# use MBQ0_5 if change sample size to n^MLBQ 
# MBQ0_5=np.array(([38,15,3],[77,30,5],[153,60,10]))


# open the file in the write mode
PE_Est = open('S3S4.csv', 'w')


# create the csv writer
writer = csv.writer(PE_Est)


# repeat 100 runs
for i in range(100):
    # set a random seed
    print("i="+str(i))
    np.random.seed(seed=i)

    # Iterate for different budget sizes
    for n_level in range(3):

        Est = MLBQ(L=3, N=MLN[n_level,:], solver=PE_solver, sampler=PE_S3S4sampler, hyper=torch.tensor([2.]), kernel=Matern05, KM=KM_Matern05, CalIE=True, IE=IE_Matern05, sum=False, nuggetL=[1e-5,1e-5,1e-5], method="LBFGS", lrL=[0.02]*3, CalMLMC=True)

        # save BQ estimates for each level
        for l in range(3):
            the_row = Est[l,:]
            writer.writerow(the_row)


# close the file

PE_Est.close()
