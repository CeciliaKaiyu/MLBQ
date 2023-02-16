# visualize the result

from ODE.ODE_app.Get_MultiEst import *

#--------------------Step 1 : Prepare the data ------------------------------------

# Get results for MLBQ with n^MLBQ and regular grid
S1data = pd.read_csv(r"S1.csv", header=None)
S1data = S1data.to_numpy()
S1E=S1data[:,0]--1/12

# Get results for MLBQ with n^MLMC and regular grid
S2data = pd.read_csv(r"S2.csv", header=None)
S2data = S2data.to_numpy()
S2E=S2data[:,0]--1/12

# Get results for MLBQ and MLMC with IID samples
S3S4data = pd.read_csv(r"S3S4.csv", header=None)

S3E=Get_Output(S3S4data,types=0,N_budget=3,N_runs=100,L=3,l=3)--1/12
S3V=Get_Output(S3S4data,types=1,N_budget=3,N_runs=100,L=3,l=3)
S3El1=Get_Output(S3S4data,types=0,N_budget=3,N_runs=100,L=3,l=2)--1/12

# Get result for setting 4
S4E=Get_Output(S3S4data,types=2,N_budget=3,N_runs=100,L=3,l=3)--1/12


#compute the covarage probability
Cov_ProbS3=Coverage_Prob(Acc_Est=0, E=S3E,V=S3V,CI=np.array([0.2,0.4,0.6,0.8]))



#-------------------- Step 2 : visualization------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


sns.set_style("ticks")
plt.figure(figsize=(11.5,4))


Sample_Cons = np.array(["0.376","0.753","1.503"])
N_runs=100
S1="MLBQ $n^{MLBQ}$ Grid"
S2="MLBQ $n^{MLMC}$ Grid"
S3="MLBQ $n^{MLMC}$ IID"
S4="MLMC $n^{MLMC}$ IID"


# Plot - 1 - Boxplot of Absolute Error

Err=pd.DataFrame(columns=["Budget T (s)","Estimator","Absolute Error"])

for i in range(np.size(Sample_Cons)):
    data = {
        "Budget T (s)": np.repeat(Sample_Cons[i], N_runs*2+2),
        "Estimator": np.concatenate((np.array([S1]), np.array([S2]),np.array(np.repeat([S3, S4], N_runs)))),
        "Absolute Error": np.concatenate(
            ( np.array([np.abs(S1E[i]), np.abs(S2E[i])]), np.abs(S3E[i,0:]),np.abs(S4E[i,0:])))
    }
    df=pd.DataFrame(data)
    Err=Err.append(df, ignore_index=True)

colors = ["tab:blue","tab:orange","tab:green","tab:red"]
sns.set_palette(sns.color_palette(colors))
plt.subplot(1, 3, 1)
g1=sns.boxplot(x="Budget T (s)", y="Absolute Error", hue="Estimator", data=Err)
for line in g1.get_lines()[4::24]:
    line.set_color("tab:blue")
for line in g1.get_lines()[10::24]:
    line.set_color("tab:orange")
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))



# Plot - 2 - Line Plot

#Absolute error and empirical confidence interval for 100 runs when different estimator and different sample size is used with level 0-1 / 0-2

levelsS3=pd.DataFrame(columns=["Budget T (s)","Estimator","Absolute Error","levels"])

for i in range(np.size(Sample_Cons)):
    for j in range(1,3):
        if j==1:
            data = {
            "Budget T (s)": np.repeat(Sample_Cons[i], N_runs*2),
            "Estimator": np.repeat([ S3,S4], N_runs),
            "Absolute Error": np.concatenate(
            (  np.abs(S3El1[i,0:]),np.abs(S4El1[i,0:]))),
            "levels": np.repeat(j+1, N_runs*2)
            }
            df=pd.DataFrame(data)
            levelsS3=levelsS3.append(df, ignore_index=True)
        if j==2:
            data = {
            "Budget T (s)": np.repeat(Sample_Cons[i], N_runs*2),
            "Estimator": np.repeat([ S3,S4], N_runs),
            "Absolute Error": np.concatenate(
            (  np.abs(S3E[i,0:]),np.abs(S4E[i,0:]))),
            "levels": np.repeat(j+1, N_runs*2)
            }
            df=pd.DataFrame(data)
            levelsS3=levelsS3.append(df, ignore_index=True)


colors = ["tab:green", "tab:red"]
sns.set_palette(sns.color_palette(colors))
plt.subplot(1, 3, 2)
g2=sns.lineplot(x="Budget T (s)", y="Absolute Error",style="levels",hue="Estimator",markers=True, data=levelsS3)
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))



# Plot - 3 - Calibration Plot


refdiag={
    "Budget T (s)": np.repeat(Sample_Cons[0], 6),
    "Estimator": np.repeat(["Ref"], 6),
    "Credible Level": np.array(["0", "0.2", "0.4", "0.6", "0.8", "1"]),
    "Coverage Probability": np.array([0, 0.2, 0.4, 0.6, 0.8, 1])

}
refdf = pd.DataFrame(refdiag)


Calib=pd.DataFrame(columns=["Budget T","Estimator","Credible Level","Coverage Probability"])

for i in range(3):
    data = {
        "Budget T (s)": np.repeat(Sample_Cons[i], 6),
        "Credible Level":np.tile(np.array(["0", "0.2", "0.4", "0.6", "0.8", "1"]), 1),
        "Coverage Probability": np.concatenate(
            ([0.],Cov_ProbS3[i,0:],[1.]) )
    }
    df=pd.DataFrame(data)
    Calib=Calib.append(df, ignore_index=True)

del Calib["Estimator"]

plt.subplot(1, 3, 3)
cl=np.array(["0", "0.2", "0.4", "0.6", "0.8", "1"])
g3=sns.lineplot(data=refdf, x="Credible Level", y="Coverage Probability",color="black", label='_nolegend_')
line1 = plt.plot(cl,np.concatenate(([0.],Cov_ProbS3[0,0:],[1.])), label='0.376',color="green",linestyle='solid',marker="o",markersize=3)
line2 = plt.plot(cl,np.concatenate(([0.],Cov_ProbS3[1,0:],[1.])), label='0.751',color="green",linestyle='dashed',marker="X",markersize=3)
line3 = plt.plot(cl,np.concatenate(([0.],Cov_ProbS3[2,0:],[1.])), label='1.503',color="green",linestyle='dotted',marker="s",markersize=3)
plt.xlabel('Credible Level', fontsize=18)
plt.ylabel('Coverage Probability', fontsize=18)
plt.legend(title='Budget T (s)')



plt.tight_layout()
plt.savefig("PE_Plot.pdf",dpi=600)
plt.show()


















