# visualize the result

from ODE.ODE_app.Get_MultiEst import *

#--------------------Step 1 : Prepare the data ------------------------------------

# BQ
BQdata = pd.read_csv(r"ODE_BQ.csv", header=None)

BQE = Get_Output(BQdata,types=0,N_budget=3,N_runs=100,L=1,l=1)--3.417076
BQV = Get_Output(BQdata,types=1,N_budget=3,N_runs=100,L=1,l=1)

# estimation with optimal sample size of MLBQ with Matern 2.5 MLMC and IID samples
MCdata = pd.read_csv(r'ODE_MC.csv', header=None)
# estimation with optimal sample size of MLMC and IID samples
MCbigdata = pd.read_csv(r"ODE_large.csv", header=None)
# estimation with optimal sample size of MLBQ with SE kernel MLMC and IID samples
Gaussdata = pd.read_csv(r'ODE_MC_withGauss.csv', header=None)

MLBMCE = Get_Output(MCdata,types=0,N_budget=3,N_runs=100,L=3,l=3)--3.417076
MLBMCEl1 = Get_Output(MCdata,types=0,N_budget=3,N_runs=100,L=3,l=2)--3.417076
MLBMCV = Get_Output(MCdata,types=1,N_budget=3,N_runs=100,L=3,l=3)

MLMCE = Get_Output(MCdata,types=2,N_budget=3,N_runs=100,L=3,l=3)--3.417076
MLMCEl1 = Get_Output(MCdata,types=2,N_budget=3,N_runs=100,L=3,l=2)--3.417076

# estimation with optimal sample size of MLMC and Big budget
MLMCbigE = Get_Output(MCbigdata,types=0,N_budget=2,N_runs=100,L=3,l=3)--3.417076

# estimation with optimal sample size of MLMC and QMC point
QMCdata = pd.read_csv(r"ODE_Halton.csv", header=None)
QMCE = Get_Output(QMCdata,types=0,N_budget=3,N_runs=100,L=1,l=1)--3.417076

# estimation with optimal sample size of MLMC and LHS point
LHSdata = pd.read_csv(r"ODE_LHS.csv", header=None)
LHSE = Get_Output(LHSdata,types=0,N_budget=3,N_runs=100,L=1,l=1)--3.417076

# estimation with optimal sample size of MLMC and SE kernel
GaussE = Get_Output(Gaussdata,types=0,N_budget=3,N_runs=100,L=1,l=1)--3.417076


#compute the covarage probability
Cov_ProbMLBMC=Coverage_Prob(Acc_Est=0, E=MLBMCE,V=MLBMCV,CI=np.array([0.2,0.4,0.6,0.8]))
Cov_ProbBQ=Coverage_Prob(Acc_Est=0, E=BQE,V=BQV,CI=np.array([0.2,0.4,0.6,0.8]))



#-------------------- Step 2 : visualization------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_style("ticks")


plt.figure(figsize=(18,6))
plt.rcParams['font.size'] = '15'

Sample_Cons = ["0.303","0.910","1.517"]
BigSample_Cons = ["30.347","151.736"]
N_runs=100

# Plot - 1 - Boxplot of Absolute Error

Err=pd.DataFrame(columns=["Budget T (s)","Estimator","Absolute Error"])

for i in [0,2]:
    data = {
        "Budget T (s)": np.repeat(Sample_Cons[i], N_runs*6),
        "Estimator": np.repeat([ "MLBQ 2.5 QMC","MLBQ 2.5 LHS","MLBQ SE IID","MLBQ 2.5 IID",  "BQ 2.5 IID", "MLMC"], N_runs),
        "Absolute Error": np.concatenate(
            ( np.abs(QMCE[i,0:]), np.abs(LHSE[i,0:]), np.abs(MLBMCE[i,0:]),np.abs(GaussE[i,0:]),np.abs(BQE[i,0:]),np.abs(MLMCE[i,0:])))
    }
    df=pd.DataFrame(data)
    Err=Err.append(df, ignore_index=True)

for i in range(2):
    data = {
        "Budget T (s)": np.repeat(BigSample_Cons[i], N_runs),
        "Estimator": np.repeat(["MLMC"], N_runs),
        "Absolute Error": np.abs(MLMCbigE[i,0:])
    }
    df=pd.DataFrame(data)
    Err=Err.append(df, ignore_index=True)

colors = ["tab:blue","tab:orange","tab:green","tab:cyan","tab:purple","tab:red"]
sns.set_palette(sns.color_palette(colors))
plt.subplot(1, 3, 1)
g1=sns.boxplot(x="Budget T (s)", y="Absolute Error", hue="Estimator", data=Err)
g1.set(yscale="log")
plt.xlabel('Budget T (s)', fontsize=18)
plt.ylabel('Absolute Error', fontsize=18)


# Plot - 2 - Line Plot

#Absolute error and empirical confidence interval for 100 runs when different estimator and different sample size is used with level 0-1 / 0-2

levelsS3=pd.DataFrame(columns=["Budget T (s)","Estimator","Absolute Error","L"])

for i in [0,2,]:
    for j in range(1,3):
        if j==1:
            data = {
            "Budget T (s)": np.repeat(Sample_Cons[i], N_runs*2),
            "Estimator": np.repeat([ "MLBQ 2.5 IID","MLMC"], N_runs),
            "Absolute Error": np.concatenate(
            (  np.abs(MLBMCEl1[i,0:]),np.abs(MLMCEl1[i,0:]))),
            "L": np.repeat(j, N_runs*2)
            }
            df=pd.DataFrame(data)
            levelsS3=levelsS3.append(df, ignore_index=True)
        if j==2:
            data = {
            "Budget T (s)": np.repeat(Sample_Cons[i], N_runs*2),
            "Estimator": np.repeat([ "MLBQ 2.5 IID","MLMC"], N_runs),
            "Absolute Error": np.concatenate(
            (  np.abs(MLBMCE[i,0:]),np.abs(MLMCE[i,0:]))),
            "L": np.repeat(j, N_runs*2)
            }
            df=pd.DataFrame(data)
            levelsS3=levelsS3.append(df, ignore_index=True)


colors = ["tab:green", "tab:red"]
sns.set_palette(sns.color_palette(colors))
plt.subplot(1, 3, 2)
g2=sns.lineplot(x="Budget T (s)", y="Absolute Error",style="L",hue="Estimator",markers=True, data=levelsS3)
g2.set(ylim=(-0.0, 0.35))
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.xlabel('Budget T (s)', fontsize=18)
plt.ylabel('Absolute Error', fontsize=18)


# Plot - 3 - Calibration Plot

# for drawing the diagnol reference curve
refdiag={
    "Budget T (s)": np.repeat(Sample_Cons[0], 6),
    "Estimator": np.repeat(["Ref"], 6),
    "Credible Level": np.array([0, 0.2, 0.4, 0.6, 0.8, 1]),
    "Coverage Probability": np.array([0, 0.2, 0.4, 0.6, 0.8, 1])

}
refdf = pd.DataFrame(refdiag)

#Coverage Probability for 100 runs when different estimator and different sample size is used with level 0-2
Calib=pd.DataFrame(columns=["Budget T (s)","Estimator","Credible Level","Coverage Probability"])


for i in [0,2]:
    data = {
        "Budget T (s)": np.repeat(Sample_Cons[i], 12),
        "Estimator": np.repeat(["MLBQ 2.5 IID", "BQ 2.5 IID"],6),
        "Credible Level":np.tile(np.array([0,0.2,0.4,0.6,0.8,1]), 2),
        "Coverage Probability": np.concatenate(
            ([0.],Cov_ProbMLBMC[i,0:],[1.], [0.],Cov_ProbBQ[i,0:],[1.]  ))
    }
    df=pd.DataFrame(data)
    Calib=Calib.append(df, ignore_index=True)

colors = ["tab:green","tab:purple"]
sns.set_palette(sns.color_palette(colors))

plt.subplot(1, 3, 3)
sns.lineplot(data=refdf, x="Credible Level", y="Coverage Probability",color="black")
g3=sns.lineplot(data=Calib, x="Credible Level", y="Coverage Probability",hue="Estimator",markers=True,style="Budget T (s)",color="green")
plt.xlabel("Credible Level", fontsize=18)
plt.ylabel("Coverage Probability", fontsize=18)


plt.tight_layout()
plt.savefig("ODE_Plot.pdf",dpi=600)
plt.show()

















