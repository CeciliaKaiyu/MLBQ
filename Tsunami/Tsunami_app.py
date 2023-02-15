# Implementation of MLBQ and MLMC under budget limit=1200s

import pandas as pd
import numpy as np
import csv
from src.BQ import *
from Tsunami.Tsunami_src.KM_IE import *
from Tsunami.Tsunami_src.Kernels import *


# Read inner integral data
data = pd.read_csv(r"../Tsunami_Data/Inner_32.csv", header=None)

N_runs=20

# energy flux
alleta2_RS = data[[0]]
alleta2_RS = alleta2_RS.to_numpy()
alleta2_RS =np.reshape(alleta2_RS,(N_runs,92))
alleta2_RS = torch.tensor(alleta2_RS)
alleta2_RS = alleta2_RS*9.8

# momentum flux
allHV2_RS = data[[1]]
allHV2_RS = allHV2_RS.to_numpy()
allHV2_RS = np.reshape(allHV2_RS,(N_runs,92))
allHV2_RS = torch.tensor(allHV2_RS)

# Create a np array to save results
Est = np.zeros(8)
# open the file in the write mode
mu_Est = open('Est_1200.csv', 'w')
# create the csv writer
writer = csv.writer(mu_Est)



meshl_start = [0,2,6,14,30]
meshl_end = [2,6,14,30,62]

for j in range(1,21):
    # read \omega
    Path="../Tsunami_Data/inputs32/"+str(j)+"/input_MC.txt"
    Paras = np.loadtxt(Path, delimiter=',', skiprows=0,
                       dtype=float)

    # prepare data for jth run
    Paras = torch.tensor(Paras)
    eta2_RS = alleta2_RS[j-1,0:]
    HV2_RS = allHV2_RS[j-1,0:]
    runs = 0

    # compute increments
    for i in range(1,6):

        # take \omega at a level
        N_l = 2**i
        X = Paras[meshl_start[i-1]:meshl_end[i-1]]

        # take flux at a fine level
        eta2_f = eta2_RS[runs:(runs+N_l)]
        HV2_f = HV2_RS[runs:(runs+N_l)]

        if i <5:

            # take flux at a coarse level

            eta2_c = eta2_RS[(runs+N_l):(runs+2*N_l)]
            HV2_c = HV2_RS[(runs+N_l):(runs+2*N_l)]

            runs = runs+2*N_l

        else:
            # take flux at a coarse level

            eta2_c = torch.zeros(N_l)
            HV2_c = torch.zeros(N_l)


        # compute diff between coarse level and fine level
        eta2_diff = eta2_f - eta2_c
        HV2_diff = HV2_f - HV2_c

        # compute batch size
        size = torch.tensor([X.size()[0], 5])
        bs = torch.min(size)

        # compute integral over \omega at the level

        # compute BQ esimtates

        E,V = BQ(X=X, Y=eta2_diff, Gram=Gram, hyper=torch.tensor([2.,2.,2.]), method="Adam", nugget=0, CalIE=True, \
                      kernel=Matern25, epoch=700, batch_size=int(bs), lr=0.02, \
                      KM=KM_Matern25,IE=IE_Matern25)
        Est[0] = float(E)
        Est[1] = float(V)

        # compute MC esimtates

        Est[2] = float(torch.mean(eta2_diff))
        Est[3] = float(torch.var(eta2_diff) / X.size(0))

        # compute BQ esimtates

        E,V = BQ(X=X, Y=HV2_diff, Gram=Gram, hyper=torch.tensor([2.,2.,2.]), method="Adam", nugget=0, CalIE=True, \
                      kernel=Matern25, epoch=700, batch_size=int(bs), lr=0.02, \
                      KM=KM_Matern25,IE=IE_Matern25)

        Est[4] = float(E)
        Est[5] = float(V)

        # compute MC esimtates

        Est[6] = float(torch.mean(HV2_diff))
        Est[7] = float(torch.var(HV2_diff) / X.size(0))


        # write a row to the csv file
        writer.writerow(Est)


# close the file
mu_Est.close()
