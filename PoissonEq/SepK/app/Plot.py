# visualize the result

from ODE.ODE_app.Get_MultiEst import *

#--------------------Step 1 : Prepare the data ------------------------------------

# MLBQ & MLMC IID
S3S4data = pd.read_csv(r"S3S4.csv", header=None)
S3E=Get_Output(S3S4data,types=0,N_budget=3,N_runs=100,L=3,l=3)--1/12
S4E=Get_Output(S3S4data,types=2,N_budget=3,N_runs=100,L=3,l=3)--1/12

#B1
S5adata = pd.read_csv(r"S5_001.csv", header=None)
S5aE=Get_Output(S5adata,types=0,N_budget=3,N_runs=100,L=1,l=1)--1/12

#B2
S5bdata = pd.read_csv(r"S5_005.csv", header=None)
S5bE=Get_Output(S5bdata,types=0,N_budget=3,N_runs=100,L=1,l=1)--1/12

#B3
S5cdata = pd.read_csv(r"S5_01.csv", header=None)
S5cE=Get_Output(S5cdata,types=0,N_budget=3,N_runs=100,L=1,l=1)--1/12


#-------------------- Step 2 : visualization------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("ticks")
plt.figure(figsize=(6,5.5))


plt.rcParams['font.size'] = '17'

Sample_Cons = np.array(["0.376","0.751","1.503"])
N_runs=100
S3="MLBQ $n^{MLMC}$ IID"
S4="MLMC $n^{MLMC}$ IID"
S5a="SK 0.01 BQ $n^{MLMC}$ IID"
S5b="SK 0.05 BQ $n^{MLMC}$ IID"
S5c="SK 0.1 BQ $n^{MLMC}$ IID"

# Plot - Boxplot of Absolute Error


Err=pd.DataFrame(columns=["Budget T (s)","Estimator","Absolute Error"])

for i in range(np.size(Sample_Cons)):
    data = {
        "Budget T (s)": np.repeat(Sample_Cons[i], N_runs*5),
        "Estimator": np.repeat([S3, S4, S5a, S5b, S5c], N_runs),
        "Absolute Error": np.concatenate(
            ( np.abs(S3E[i,0:]),np.abs(S4E[i,0:]),np.abs(S5aE[i,0:]),np.abs(S5bE[i,0:]),np.abs(S5cE[i,0:])))
    }
    df=pd.DataFrame(data)
    Err=Err.append(df, ignore_index=True)


colors = ["tab:green","tab:red","tab:blue","tab:orange","tab:purple"]
sns.set_palette(sns.color_palette(colors))
g1=sns.boxplot(x="Budget T (s)", y="Absolute Error", hue="Estimator", data=Err)
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.xlabel('Budget T (s)')
plt.ylabel('Absolute Error')


plt.tight_layout()
plt.savefig("S5_Plot.pdf",dpi=600)
plt.show()


















