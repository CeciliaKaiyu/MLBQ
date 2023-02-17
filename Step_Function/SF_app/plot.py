# visualize the result

from ODE.ODE_app.Get_MultiEst import *
from Step_Function.SF_src.f_l import *
import pandas as pd

#--------------------Step 1 : Prepare the data ------------------------------------


# MLBQ Matern / SE + iid and MLMC
A3data = pd.read_csv(r"Est_A3.csv", header=None)

A3E1=Get_Output(A3data,types=0,N_budget=2,N_runs=100,L=3,l=3)-5.
A3E2=Get_Output(A3data,types=2,N_budget=2,N_runs=100,L=3,l=3)-5.
A3E3=Get_Output(A3data,types=4,N_budget=2,N_runs=100,L=3,l=2)-5.

# Matern / SE kernel + bad design
A5bdata = pd.read_csv(r"Est_A5b.csv", header=None)
A5bE1=Get_Output(A5bdata,types=0,N_budget=2,N_runs=100,L=3,l=3)-5.
A5bE2=Get_Output(A5bdata,types=2,N_budget=2,N_runs=100,L=3,l=3)-5.




#-------------------- Step 2 : visualization------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


sns.set_style("ticks")
plt.figure(figsize=(10,6))

plt.rcParams['font.size'] = '16'

Sample_Cons = np.array(["0.002","0.004"])
N_runs=100
S1="MLBQ 0.5 bad"
S2="MLBQ SE bad"
S3="MLBQ 0.5 IID"
S4="MLBQ SE IID"
S5="MLMC"

# Plot - 1 - Boxplot of Absolute Error

Err=pd.DataFrame(columns=["Budget T (s)","Estimator","Absolute Error"])

for i in [0,1]:
    mydata = {
        "Budget T (s)": np.repeat(Sample_Cons[i], N_runs*5),
        "Estimator": np.repeat([S1, S2, S3, S4, S5], N_runs),
        "Absolute Error": np.concatenate(
            ( np.abs(A5bE1[i,0:]),np.abs(A5bE2[i,0:]), np.abs(A3E1[i,0:]),np.abs(A3E2[i,0:]),np.abs(A3E3[i,0:])))
    }
    df=pd.DataFrame(mydata)
    Err=Err.append(df, ignore_index=True)


colors = ["tab:pink","tab:blue","tab:green","tab:orange","tab:red"]
sns.set_palette(sns.color_palette(colors))
plt.subplot(1, 2, 2)
g1=sns.boxplot(x="Budget T (s)", y="Absolute Error", hue="Estimator", data=Err)
g1.set(yscale="log")
plt.xlabel('Budget T (s)', fontsize=16)
plt.ylabel('Absolute Error', fontsize=16)



#Plot - 2 f_0,f_1,f_2
hl=np.array([2,0.5,0.1])
ax = plt.subplot(1, 2, 1)
myx=np.linspace(0, 10.0, num=500)
plt.plot(myx, f_l(hl[0],myx),lw=3, label='f_0(\u03C9)',color="black")
plt.plot(myx, f_l(hl[1],myx),lw=3, label='f_1(\u03C9)',color="blue")
plt.plot(myx, f_l(hl[2],myx),lw=3, label='f_2(\u03C9)',color="green")
plt.xlabel("\u03C9")
plt.legend()

plt.tight_layout()
plt.savefig("step_Plot.pdf",dpi=600)
plt.show()


















