#Matern / SE + iid
from Step_Function.SF_src.f_l import *
from Step_Function.SF_src.Kernels import *
from Step_Function.SF_src.KM_IE import *
from src.BQ import *
import numpy as np
import csv


# Create a np array to save results
Est = np.zeros(6)
# open the file in the write mode
mu_Est = open('Est_A3.csv', 'w')
# create the csv writer
writer = csv.writer(mu_Est)

#define sample size
MLN=np.array(([37,8,2],[74,15,4]))

hl=np.array([2,0.5,0.1])

# repeat 100 runs
for i in range(100):
    # set a random seed
    print("i="+str(i))
    np.random.seed(seed=i+2)

    # Iterate for different budget sizes
    for n_level in range(np.shape(MLN)[0]):
        for l in range(3):
            a = np.random.uniform(0, 10., int(MLN[n_level,l]))
            if l > 0:
                print(l)
                h = hl[l - 1:l + 1]
                uf = f_l(h[0],a)
                uc = f_l(h[1],a)

            else:
                print(l)
                h = [1, hl[0]]
                uf = f_l(h[0], a)
                uc = np.zeros(int(MLN[n_level,l]))

            X = torch.from_numpy(a)
            Y = torch.from_numpy(uf-uc)

            Est[4] = float(torch.mean(Y))
            Est[5] = float(torch.var(Y))

            hyper=torch.ones(1)*2.

            E,V = BQ(X=X, Y=Y, Gram=Gram, hyper=hyper, kernel=Matern05, KM=KM_Matern05, CalIE=True, IE=IE_Matern05, nugget=1e-5, method="LBFGS",  max_iter=20)

            Est[0]=float(E)
            Est[1]=float(V)

            E, V = BQ(X=X, Y=Y, Gram=Gram, hyper=hyper, kernel=Gauss, KM=KM_Gauss, CalIE=True, IE=IE_Gauss, nugget=1e-3, method="Adam")

            Est[2] = float(E)
            Est[3] = float(V)


            # write a row to the csv file
            writer.writerow(Est)


# close the file
mu_Est.close()
