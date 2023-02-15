#Visualize the results of tusnami example
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#--------------------Step 1 : Prepare the data ------------------------------------
# prepare the results for visualization
from ODE.ODE_app.Get_MultiEst import *

# Read data obtained under different budget limit from files
# under budget limit=1200
Est32 = pd.read_csv(r"Est_1200.csv", header=None)
Est32 = Est32.to_numpy()
Est32=Est32[:,[0,1,2,3,4,5,6,7]]
# under budget limit=6000
Est160 = pd.read_csv(r"Est_6000.csv", header=None)
Est160 = Est160.to_numpy()
Est160=Est160[:,[0,1,2,3,4,5,6,7]]
# under budget limit=12000
Est320 = pd.read_csv(r"Est_12000.csv", header=None)
Est320 = Est320.to_numpy()
Est320=Est320[:,[0,1,2,3,4,5,6,7]]
# ref true
Est = pd.read_csv(r"Ref.csv", header=None)


# reshape results
N_runs=20
N_cons=3

Est32 = np.reshape(Est32,(N_runs,5,8))
Est160 = np.reshape(Est160,(N_runs,5,8))
Est320 = np.reshape(Est320,(N_runs,5,8))
Est = Est.to_numpy()


# estimates for energy flux
eta32_E = np.sum(Est32[0:,0:,[0,2]],axis=1)
eta32_V = np.sum(Est32[0:,0:,[1,3]],axis=1)

eta160_E = np.sum(Est160[0:,0:,[0,2]],axis=1)
eta160_V = np.sum(Est160[0:,0:,[1,3]],axis=1)

eta320_E = np.sum(Est320[0:,0:,[0,2]],axis=1)
eta320_V = np.sum(Est320[0:,0:,[1,3]],axis=1)


eta_E = np.sum(Est[0:,0],axis=0)

# estimates for momentum flux
HV232_E = np.sum(Est32[0:,0:,[4,6]],axis=1)
HV232_V = np.sum(Est32[0:,0:,[5,7]],axis=1)

HV2160_E = np.sum(Est160[0:,0:,[4,6]],axis=1)
HV2160_V = np.sum(Est160[0:,0:,[5,7]],axis=1)

HV2320_E = np.sum(Est320[0:,0:,[4,6]],axis=1)
HV2320_V = np.sum(Est320[0:,0:,[5,7]],axis=1)

HV2_E = np.sum(Est[0:,2],axis=0)

# error
Err1_32 = eta_E-eta32_E
Err1_160 = eta_E-eta160_E
Err1_320 = eta_E-eta320_E

#error 
Err2_32 = HV2_E-HV232_E
Err2_160 = HV2_E-HV2160_E
Err2_320 = HV2_E-HV2320_E


# Compute covarage probability for energy flux
etaMatern25E = np.zeros((N_cons,N_runs))
etaMatern25E[0,0:] = eta32_E[0:,0]
etaMatern25E[1,0:] = eta160_E[0:,0]
etaMatern25E[2,0:] = eta320_E[0:,0]

etaMatern25V = np.zeros((N_cons,N_runs))
etaMatern25V[0,0:] = eta32_V[0:,0]
etaMatern25V[1,0:] = eta160_V[0:,0]
etaMatern25V[2,0:] = eta320_V[0:,0]

Cov_Prob1=Coverage_Prob(Acc_Est=eta_E, E=etaMatern25E,V=etaMatern25V,CI=np.array([0.2,0.4,0.6,0.8]))


#Compute covarage probability for momentum flux
HV2Matern25E = np.zeros((N_cons,N_runs))
HV2Matern25E[0,0:] = HV232_E[0:,0]
HV2Matern25E[1,0:] = HV2160_E[0:,0]
HV2Matern25E[2,0:] = HV2320_E[0:,0]

HV2Matern25V = np.zeros((N_cons,N_runs))
HV2Matern25V[0,0:] = HV232_V[0:,0]
HV2Matern25V[1,0:] = HV2160_V[0:,0]
HV2Matern25V[2,0:] = HV2320_V[0:,0]

Cov_Prob3=Coverage_Prob(Acc_Est=HV2_E, E=HV2Matern25E,V=HV2Matern25V,CI=np.array([0.2,0.4,0.6,0.8]))



#-------------------- Step 2 : visualization------------------------------------

sns.set_style("ticks")


plt.figure(figsize=(10.,9))
plt.rcParams['font.size'] = '18'
Sample_Cons = np.array(["1200", "6000", "12000"])
N_runs=20
Method=["MLBQ IID",  "MLMC"]

# Plot - 1 - Boxplot of Absolute Error for Energy Flux
Error1DF=pd.DataFrame(columns=["Budget T (s)","Estimator","log (Absolute Error)"])

for i in range(2):
    data = {
        "Budget T (s)": np.repeat(Sample_Cons, N_runs),
        "Estimator": np.repeat(Method[i], N_runs*np.shape(Sample_Cons)[0]),
        "Absolute Error": np.concatenate(
            ( np.abs(Err1_32[0:,i]),np.abs(Err1_160[0:,i]),np.abs(Err1_320[0:,i])))
    }
    df=pd.DataFrame(data)
    Error1DF=Error1DF.append(df, ignore_index=True)


colors = ["tab:green","tab:red"]

sns.set_palette(sns.color_palette(colors))


plt.subplot(2, 2, 1)
plt.title("Energy")
g1=sns.boxplot(x="Budget T (s)", y="Absolute Error", hue="Estimator", data=Error1DF)
g1.set(yscale="log")
plt.ylim( (10**-2,10**4) )


# Plot - 2 - Boxplot of Absolute Error for Momentum Flux
Error2DF=pd.DataFrame(columns=["Budget T (s)","Estimator","log (Absolute Error)"])

for i in range(2):
    data = {
        "Budget T (s)": np.repeat(Sample_Cons, N_runs),
        "Estimator": np.repeat(Method[i], N_runs*np.shape(Sample_Cons)[0]),
        "Absolute Error": np.concatenate(
            ( np.abs(Err2_32[0:,i]),np.abs(Err2_160[0:,i]),np.abs(Err2_320[0:,i])))
    }
    df=pd.DataFrame(data)
    Error2DF=Error2DF.append(df, ignore_index=True)



plt.subplot(2, 2, 2)
plt.title("Momentum Flux")
g2=sns.boxplot(x="Budget T (s)", y="Absolute Error", hue="Estimator", data=Error2DF)
g2.set(yscale="log")
plt.legend([],[], frameon=False)
plt.ylim( (10**-2,10**4) )

# Plot - 3 - Calibration Plot for Energy Flux

refdiag={
    "Budget T (s)": np.repeat(Sample_Cons[1], 6),
    "Estimator": np.repeat(["Ref"], 6),
    "Credible Level": np.array(["0", "0.2", "0.4", "0.6", "0.8", "1"]),
    "Coverage Probability": np.array([0, 0.2, 0.4, 0.6, 0.8, 1])

}
refdf = pd.DataFrame(refdiag)


plt.subplot(2, 2, 3)
cl=np.array(["0", "0.2", "0.4", "0.6", "0.8", "1"])
g3=sns.lineplot(data=refdf, x="Credible Level", y="Coverage Probability",color="black",label='_nolegend_')
line1 = plt.plot(cl,np.concatenate(([0.],Cov_Prob1[0,0:],[1.])), label="1200",color="green",linestyle='solid',marker="o",markersize=3)
line2 = plt.plot(cl,np.concatenate(([0.],Cov_Prob1[1,0:],[1.])), label="6000",color="green",linestyle='dashed',marker="X",markersize=3)
line3 = plt.plot(cl,np.concatenate(([0.],Cov_Prob1[2,0:],[1.])), label="12000",color="green",linestyle='dotted',marker="s",markersize=3)
plt.legend(title='Budget T (s)',loc='upper left',)

# Plot - 4 - Calibration Plot for Momentum Flux


refdiag={
    "Budget T (s)": np.repeat(Sample_Cons[1], 6),
    "Estimator": np.repeat(["Ref"], 6),
    "Credible Level": np.array(["0", "0.2", "0.4", "0.6", "0.8", "1"]),
    "Coverage Probability": np.array([0, 0.2, 0.4, 0.6, 0.8, 1])

}
refdf = pd.DataFrame(refdiag)

plt.subplot(2, 2, 4)
g4=sns.lineplot(data=refdf, x="Credible Level", y="Coverage Probability",color="black")
line1 = plt.plot(cl,np.concatenate(([0.],Cov_Prob3[0,0:],[1.])), label="1200",color="green",linestyle='solid',marker="o",markersize=3)
line2 = plt.plot(cl,np.concatenate(([0.],Cov_Prob3[1,0:],[1.])), label="6000",color="green",linestyle='dashed',marker="X",markersize=3)
line3 = plt.plot(cl,np.concatenate(([0.],Cov_Prob3[2,0:],[1.])), label="12000",color="green",linestyle='dotted',marker="s",markersize=3)
plt.legend([],[], frameon=False)
plt.tight_layout()
plt.savefig("TsunamiPlot.pdf",dpi=300)
plt.show()


