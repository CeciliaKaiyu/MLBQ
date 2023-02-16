# MLMC with big budget multiple runs

import torch
import csv
from ODE.ODE_src.ODE_Solver import *


# determine the sample size at each level 0,1,2
MLBig=np.array(([16579,2701,308],[82984,13505,1538]))

# open the file in the write mode
ODE_Est = open('ODE_large.csv', 'w')


# create the csv writer
writer = csv.writer(ODE_Est)
Est=np.zeros(2)


# repeat 100 runs
for i in range(100):
    # set a random seed
    print("i="+str(i))
    np.random.seed(seed=i+1234)

    # try different samples sizes
    for n_level in range(2):

        for l in range(3):
            print(l)

            # IID samples
            X = ODE_Sampler(N=int(MLBig[n_level,l]))
            # fine and coarse approximation
            Pf, Pc = ODE_Solver(X, l=l)
            # compute MC estimate
            Y = torch.from_numpy(Pf - Pc)
            X = torch.from_numpy(X)
            Est[0] = float(torch.mean(Y))
            Est[1]= float(torch.var(Y)/X.size(0))

            # write a row to the csv file
            writer.writerow(Est)





# close the file

ODE_Est.close()