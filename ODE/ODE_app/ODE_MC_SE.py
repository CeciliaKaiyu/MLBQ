# MLBQ with MC sampling and MLMC - multiple runs
# SE kernel
import csv
from ODE.ODE_src.Kernels import *
from ODE.ODE_src.KM_IE import *
from ODE.ODE_src.ODE_Solver import *
from src.MLBQ import *



# define the sampler
def MC_Sampler(N):
    N= int(N)
    X = ODE_Sampler(N)
    return X


Est=np.zeros(4)

# determine the sample size at each level 0,1,2
MLN=np.array(([166,27,3],[830,135,15]))


# open the file in the write mode
ODE_Est = open('ODE_MC_withGauss.csv', 'w')
writer = csv.writer(ODE_Est)


# repeat 100 runs
for i in range(100):
    # set a random seed
    print("i="+str(i))
    np.random.seed(seed=i+1234)

    # Iterate for different budget sizes
    for n_level in range(3):

        if n_level == 0:
            my_nugget = 1e-6
        else:
            my_nugget = [1e-8,1e-9,1e-10]

        Est = MLBQ(L=3, N=MLN[n_level,:], solver=ODE_Solver, sampler=MC_Sampler, hyper=torch.tensor([2.,2.]), kernel=Gauss_ODE, KM=KM_Gauss_ODE, CalIE=False,IE=IE_Gauss_ODE, \
                     sum=True, nuggetL=[1e-6]*3, method="Adam", lrL=[0.01]*3, epochL=[20]*3, maxBatch_size=10)

        # save MLBQ estimates
        writer.writerow(Est)


# close the file

ODE_Est.close()