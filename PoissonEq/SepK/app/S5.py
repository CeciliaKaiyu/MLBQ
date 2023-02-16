## MLBQ with separable kernel

from PoissonEq.PE_app.SepK.src.kernels import *
from PoissonEq.PE_app.SepK.src.BQ_sk import *
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


# open the file in the write mode
PE_Est = open('S5_005.csv', 'w')


# create the csv writer
writer = csv.writer(PE_Est)
# Create a np array to save results
Est = np.zeros(2)


#define sample size
MLN=np.array(([67,11,1],[133,23,2],[166,46,3]))



# repeat 100 runs
for i in range(100):
    # set a random seed
    print("i="+str(i))
    np.random.seed(seed=i)

    # Iterate for different budget sizes
    for n_level in range(3):
        X0 = PE_S3S4sampler(MLN[n_level,0])
        X1 = PE_S3S4sampler(MLN[n_level, 1])
        X2 = PE_S3S4sampler(MLN[n_level, 2])
        u0f, u0c = PE_solver(X0, l=0)
        u1f, u1c = PE_solver(X1, l=1)
        u2f, u2c = PE_solver(X2, l=2)

        X0 = torch.unsqueeze(torch.tensor(X0), 1)
        X1 = torch.unsqueeze(torch.tensor(X1), 1)
        X2 = torch.unsqueeze(torch.tensor(X2), 1)

        Y0 = torch.unsqueeze(torch.tensor(u0f-u0c), 1)
        Y1 = torch.unsqueeze(torch.tensor(u1f-u1c), 1)
        Y2 = torch.unsqueeze(torch.tensor(u2f-u2c), 1)

        X=[X0,X1,X2]
        Y=[Y0,Y1,Y2]

        hyper=torch.ones(1)*2.

        E,V = BQ_sk(X=X, Y=Y, Gram=Gram, hyper=hyper, kernel=sep_kernel, KM=KM_Matern05, CalIE=True, IE=IE_Matern05, nugget=1e-6, max_iter=20)

        Est[0]=float(E)
        Est[1]=float(V)
        writer.writerow(Est)


# close the file

PE_Est.close()


