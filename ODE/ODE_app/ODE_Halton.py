# MLBQ with Halton sampling - multiple runs
# Matern 2.5 kernel
import random
import csv
from ODE.ODE_src.Kernels import *
from ODE.ODE_src.KM_IE import *
from ODE.ODE_src.MLBQsp import *
from ODE.ODE_src.ODE_Solver import *

# define the sampler
def Halton_Sampler(N):
    N=int(N)
    seed=random.randint(1,100)+N
    X = ODE_Sampler(N,method="Halton",seed=seed)
    return X

# determine the sample size at each level 0,1,2
MLN=np.array(([166,27,3],[830,135,15]))


# open the file in the write mode
ODE_Est = open('ODE_Halton.csv', 'w')


# create the csv writer
writer = csv.writer(ODE_Est)


Est=np.zeros(2)

X2 = np.random.normal(0, 1, 10000)
X2 = torch.from_numpy(X2)


#repeat 100 runs
for i in range(100):
    #set a random seed
    print("i="+str(i))
    np.random.seed(seed=i+1234)

    # Iterate for different budget sizes

    for n_level in range(np.shape(MLN)[0]):

        if n_level == 0:
            my_nugget = [1e-10]
        elif n_level == 1:
            my_nugget = [1e-8]
        else:
            my_nugget = [1e-6]

        Est = MLBQsp(L=3, N=MLN[n_level,:], X2=X2, solver=ODE_Solver, sampler=Halton_Sampler, hyper=torch.tensor([2.,2.]), kernel=Matern25_ODE, KM=KM_Matern25_ODE, IE1=IE_Matern25_ODE1, KM2=KM_Matern25_ODE2, \
                     CalMLMC=False, nuggetL=my_nugget*3, method="Adam", lrL=[0.01]*3, epochL=[50]*3, maxBatch_size=100)

        writer.writerow(Est)


# close the file

ODE_Est.close()