# BQ with IID samples
# Matern 2.5 kernel

# import ODE sampler and solver
from ODE.ODE_src.ODE_Solver import *
from ODE.ODE_src.Kernels import *
from ODE.ODE_src.KM_IE import *
from ODE.ODE_src.BQsp import *
import csv


Est=np.zeros(2)

X2 = np.random.normal(0, 1, 10000)
X2 = torch.from_numpy(X2)

# open the file in the write mode
ODE_Est = open('ODE_BQ.csv', 'w')

# create the csv writer
writer = csv.writer(ODE_Est)

#sample size
BQN=[15,75]

#repeat 100 runs
for i in range(100):
    #set a random seed
    print("i="+str(i))
    np.random.seed(seed=i+1234)

    # Iterate for different budget sizes
    for n_level in range(2):

        X = ODE_Sampler(N=int(BQN[n_level]))
        # fine and coarse approximation
        Pf, Pc = ODE_Solver(X, l=2)

        # convert np array to tensor
        Y = torch.from_numpy(Pf)
        X = torch.from_numpy(X)

        # BQ estimator
        E,V=BQsp(X=X, X2=X2, Y=Y, Gram=Gram, hyper=torch.tensor([2., 2.]), method="Adam", nugget=0, IE1=IE_Matern25_ODE1, KM2=KM_Matern25_ODE2, \
                                        kernel=Matern25_ODE, lr=0.01, KM=KM_Matern25_ODE, epoch=60, batch_size=100)

        Est[0] = float(E)
        Est[1] = float(V)

        # write a row to the csv file
        writer.writerow(Est)


# close the file

ODE_Est.close()