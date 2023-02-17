#Illustration plot

import matplotlib.pyplot as plt
from scipy.stats import norm
from src.Optimizer import *
from PoissonEq.PE_app.Illustration_PE.Illustration_src import *
from PoissonEq.PE_src.u_fc_array import *


# ------------------------------------create the solver------------------------------------
def PE_solver(X,l):

    u=0
    hl=np.array([0.5,0.2,0.005])

    if l > 0:
        print(l)
        h = hl[l - 1:l + 1]
        uf, uc = u_fc(h, X)
        uf = uf*7. + u
        uc = uc*7. + u


    else:
        print(l)
        h = [1, hl[0]]
        uf, uc = u_fc(h, X)
        uf = uf*7.+ u

    return uf, uc

# create the sampler
def PE_S1sampler(N):
    X = np.append(1/(N-1) * np.array(list(range(int(N-1)))), 1.)
    return X

#------------------------------------BQ and GP function ------------------------------------


def BQ(myx, X, Y, nugget=0):

    myx_gp = np.unique(np.append(myx, X))
    Y = torch.from_numpy(Y)
    X = torch.from_numpy(X)


    Hyper_cp = training_loop(X=X, Y=Y, Gram=Gram, hyper=torch.tensor([2.]), kernel=Matern05)
    Hyper = Hyper_cp.detach().clone()

    # Compute the amplitude with the closed form expression
    K = Matern05(X=X, hyper=Hyper) + nugget*torch.eye(X.size(0))
    Inv_K = torch.inverse(K)
    amp=Y@Inv_K@Y/Y.size(0)

    # Update kernel matrix by multiplying the amplitude
    K=amp*(K-nugget*torch.eye(X.size(0))) + nugget*torch.eye(X.size(0))
    Inv_K = torch.inverse(K)

    # compute the kernel mean and initial error with optimized hyper parameter
    my_km = amp*km(X=X, hyper=Hyper)
    # compute posterior estimate of the integral of interested with closed form solution
    E = my_km @ Inv_K @ Y
    # if initial error error has closed form solution
    my_ie = amp*ie(hyper=Hyper)
    V = my_ie - my_km @ Inv_K @ my_km

    # BQ GP

    myx_gp = torch.from_numpy(myx_gp)
    myGPmean = gp_mean(myx_gp, X, Hyper, Y)
    myGPv = amp*gp_cov(myx_gp, X, Hyper)
    myGPv = myGPv.diagonal()
    return E,V,myGPmean, myGPv,myx_gp,Hyper,amp



#------------------------------------compute BQ and GP------------------------------------
#f0, f1, f2
N=[16,11,3]
myx=np.array((range(0,201)))/200/1.
f0x,f00 = PE_solver(myx,0)
f2x,f1x = PE_solver(myx,2)

#level 0
X0 = PE_S1sampler(N[0])
Y_f0,Y_c0 = PE_solver(X0,0)
E0,V0,myGPmean0,myGPv0,myx0,Hyper0,amp0=BQ(myx, X0,Y_f0-Y_c0, nugget=0)

#level 1
X1 = PE_S1sampler(N[1])
Y_f1,Y_c1 = PE_solver(X1,1)
E1,V1,myGPmean1,myGPv1,myx1,Hyper1,amp1=BQ(myx, X1,Y_f1-Y_c1, nugget=0)

#level 2
X2 = PE_S1sampler(N[2])
Y_f2,Y_c2 = PE_solver(X2,2)
E2,V2,myGPmean2,myGPv2,myx2,Hyper2,amp2=BQ(myx, X2,Y_f2-Y_c2, nugget=0)

#BQ
X_bq = PE_S1sampler(4)
Y_fbq,Y_cbq = PE_solver(X_bq,2)

E,V,myGPmean,myGPv,myx_bq,Hyper_bq,amp_bq=BQ(myx, X_bq,Y_fbq, nugget=0)




#------------------------------------Plot------------------------------------

plt.figure(figsize=(23,9))
plt.rcParams['font.size'] = '22'
#plot
plot1 = plt.subplot2grid((2, 4), (0, 0))
plot2 = plt.subplot2grid((2, 4), (0, 1))
plot3 = plt.subplot2grid((2, 4), (0, 2))
plot4 = plt.subplot2grid((2, 4), (0, 3), rowspan=2)
plot5 = plt.subplot2grid((2, 4), (1, 0))
plot6 = plt.subplot2grid((2, 4), (1, 1))
plot7 = plt.subplot2grid((2, 4), (1, 2))


plot1.plot(myx, f2x,lw=3, label='f\u2082(\u03C9)',color="black")
plot1.plot(myx, f1x,lw=3, label='f\u2081(\u03C9)',color="mediumslateblue")
plot1.plot(myx, f0x,lw=3, label='f\u2080(\u03C9)',color="violet")

plot1.legend()
plot1.set_ylim(-1.4, 0.4)

plot5.plot(myx, f0x,lw=3, label='f\u2080(\u03C9)',color="dodgerblue")
plot6.plot(myx, f1x-f0x,lw=3, label='f\u2081(\u03C9)-f\u2080(\u03C9)',color="royalblue")
plot7.plot(myx, f2x-f1x,lw=3, label='f\u2082(\u03C9)-f\u2081(\u03C9)',color="steelblue")

lower_gp0=myGPmean0-torch.sqrt(myGPv0)*2
upper_gp0=myGPmean0+torch.sqrt(myGPv0)*2
plot5.plot(myx0,lower_gp0, color='lightblue', lw=0)
plot5.plot(myx0, myGPmean0, color='deepskyblue',label='GP Mean', lw=3)
plot5.plot(myx0,upper_gp0,color='lightblue', lw=0)
plot5.fill_between(myx0, lower_gp0, upper_gp0, color='lightblue', alpha=0.3, label='GP Mean \u00B1 2SD')
plot5.plot(X0,Y_f0-Y_c0, color='tab:blue', lw=0, marker="o",label='Samples',markersize=8)
plot5.set_xlabel("\u03C9",fontsize=22)
plot5.legend(loc=2)
plot5.set_ylim(-1.4, 0.4)

lower_gp1=myGPmean1-torch.sqrt(myGPv1)*2
upper_gp1=myGPmean1+torch.sqrt(myGPv1)*2
plot6.plot(myx1,lower_gp1, color='lightblue', lw=0)
plot6.plot(myx1, myGPmean1, color='deepskyblue', lw=3)
plot6.plot(myx1,upper_gp1,color='lightblue', lw=0)
plot6.fill_between(myx1, lower_gp1, upper_gp1, color='lightblue', alpha=0.3)
plot6.plot(X1,Y_f1-Y_c1, color='tab:blue', lw=0, marker="o",markersize=8)
plot6.set_xlabel("\u03C9",fontsize=22)
plot6.legend()


lower_gp2=myGPmean2-torch.sqrt(myGPv2)*2
upper_gp2=myGPmean2+torch.sqrt(myGPv2)*2
plot7.plot(myx2,lower_gp2, color='lightblue', lw=0)
plot7.plot(myx2, myGPmean2, color='deepskyblue', lw=3)
plot7.plot(myx2,upper_gp2,color='lightblue', lw=0)
plot7.fill_between(myx2, lower_gp2, upper_gp2, color='lightblue', alpha=0.3)
plot7.plot(X2,Y_f2-Y_c2, color='tab:blue', lw=0, marker="o",markersize=8)
plot7.set_xlabel("\u03C9",fontsize=22)
plot7.legend()


ys = np.array(range(-300,0))/300
MLBQsd=float(torch.sqrt(V0+V1+V2))
MLBQE =float(E0)+float(E1)+float(E2)
plot4.plot(norm.pdf(ys, float(E), float(torch.sqrt(V))), ys, color='springgreen', label='BQ',lw=2)
plot4.plot(norm.pdf(ys, MLBQE, MLBQsd), ys, color='blue', label='MLBQ',lw=2)
plot4.set_ylim(-0.98, -0.02)
plot4.axhline(y=-1/12*7, color='dimgray', linestyle='-',label='\u03A0[f]',lw=2)
plot4.set_xlabel("Probability Density",fontsize=22)
plot4.legend()



lower_gp=myGPmean-torch.sqrt(myGPv)*2
upper_gp=myGPmean+torch.sqrt(myGPv)*2
plot2.plot(myx_bq,lower_gp, color='lightblue', lw=0)
plot2.plot(myx_bq, myGPmean, color='limegreen',label="BQ's GP Mean", lw=3)
plot2.plot(myx_bq,upper_gp,color='lightblue', lw=0)
plot2.fill_between(myx_bq, lower_gp, upper_gp, color='lightblue', alpha=0.3)
plot2.plot(X_bq,Y_fbq, color='tab:blue', lw=0, marker="o",markersize=8)
plot2.plot(myx, f2x,lw=3, label='f\u2082(\u03C9)',color="black")
plot2.set_ylim(-1.4, 0.4)
plot2.legend()


myx=torch.from_numpy(myx)
Y_f0 = torch.from_numpy(Y_f0)
Y_c0 = torch.from_numpy(Y_c0)
X0 = torch.from_numpy(X0)
Y_f1 = torch.from_numpy(Y_f1)
Y_c1 = torch.from_numpy(Y_c1)
X1 = torch.from_numpy(X1)
Y_f2 = torch.from_numpy(Y_f2)
Y_c2 = torch.from_numpy(Y_c2)
X2 = torch.from_numpy(X2)
myMLGPv=amp0*gp_cov(myx,X0,Hyper0).diagonal() + amp1*gp_cov(myx,X1,Hyper1).diagonal() + amp2*gp_cov(myx,X2,Hyper2).diagonal()
myMLGPmean=gp_mean(myx,X0,Hyper0,Y_f0-Y_c0) + gp_mean(myx,X1,Hyper1,Y_f1-Y_c1) + gp_mean(myx,X2,Hyper2,Y_f2-Y_c2)
lower_gp=myMLGPmean-torch.sqrt(myMLGPv)*2
upper_gp=myMLGPmean+torch.sqrt(myMLGPv)*2
plot3.plot(myx,lower_gp, color='lightblue', lw=0)
plot3.plot(myx, myMLGPmean, color='tab:blue',label="MLBQ's GP Mean", lw=3)
plot3.plot(myx,upper_gp,color='lightblue', lw=0)
plot3.fill_between(myx, lower_gp, upper_gp, color='lightblue', alpha=0.3)
plot3.plot(myx, f2x,lw=3, label='f\u2082(\u03C9)',color="black")
plot3.set_ylim(-1.4, 0.4)
plot3.legend()

plt.tight_layout()
plt.savefig("Illustration.pdf",dpi=600)
plt.show()

