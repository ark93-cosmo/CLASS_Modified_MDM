#%matplotlib inline
import numpy as np
import matplotlib
matplotlib.use('agg')
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import matplotlib.cm as cmap
import matplotlib.lines as mlines

from scipy.interpolate import interp1d

###Interpolation function
def log_interp1d(xx, yy, kind='linear'):
    logx = np.log10(xx)
    logy = np.log10(yy)
    lin_interp = interp1d(logx, logy, kind=kind)
    log_interp = lambda zz: np.power(10.0, lin_interp(np.log10(zz)))
    return log_interp

###GR quantities
#Pm
pk_z0_GR = np.loadtxt('output/GR_z1_pk.dat')
pk_z1_GR = np.loadtxt('output/GR_z2_pk.dat')
pk_z2_GR = np.loadtxt('output/GR_z3_pk.dat')
k_GR = pk_z0_GR[:,0]
Pm_z0_GR = pk_z0_GR[:,1]
Pm_z1_GR = pk_z1_GR[:,1]
Pm_z2_GR = pk_z2_GR[:,1]

#CMB
CMB_GR = np.loadtxt('output/GR_cl.dat')
l_GR = CMB_GR[:,0]
Cl_TT_GR = CMB_GR[:,1]
Cl_EE_GR = CMB_GR[:,2]
Cl_TE_GR = CMB_GR[:,3]
#Ptheta
transfer_theta_tot_z0_GR = np.loadtxt('output/GR_z1_tk.dat', usecols=(12,))
transfer_theta_tot_z1_GR = np.loadtxt('output/GR_z2_tk.dat', usecols=(12,))
transfer_theta_tot_z2_GR = np.loadtxt('output/GR_z3_tk.dat', usecols=(12,))


interp_Pm_z0_GR = log_interp1d(k_GR,Pm_z0_GR)
interp_Pm_z1_GR = log_interp1d(k_GR,Pm_z1_GR)
interp_Pm_z2_GR = log_interp1d(k_GR,Pm_z2_GR)
interp_Ptheta_z0_GR = log_interp1d(k_GR,transfer_theta_tot_z0_GR**2.)
interp_Ptheta_z1_GR = log_interp1d(k_GR,transfer_theta_tot_z1_GR**2.)
interp_Ptheta_z2_GR = log_interp1d(k_GR,transfer_theta_tot_z2_GR**2.)
#Theta and delta @k=1Mpc^{-1}
tau_GR=np.loadtxt('output/GR_perturbations_k2_s.dat', usecols=(0,))
theta_cdm_GR=np.loadtxt('output/GR_perturbations_k2_s.dat', usecols=(16,))
delta_cdm_GR=np.loadtxt('output/GR_perturbations_k2_s.dat', usecols=(15,))
interp_theta_cdm_GR = log_interp1d(tau_GR,theta_cdm_GR)
interp_delta_cdm_GR = log_interp1d(tau_GR,abs(delta_cdm_GR))


###MDMexact quantities
#Pm
pk_z0_MDMexact = np.loadtxt('output/MDMexact_z1_pk.dat')
pk_z1_MDMexact = np.loadtxt('output/MDMexact_z2_pk.dat')
pk_z2_MDMexact = np.loadtxt('output/MDMexact_z3_pk.dat')
k_MDMexact = pk_z0_MDMexact[:,0]
Pm_z0_MDMexact = pk_z0_MDMexact[:,1]
Pm_z1_MDMexact = pk_z1_MDMexact[:,1]
Pm_z2_MDMexact = pk_z2_MDMexact[:,1]

#CMB
CMB_MDMexact = np.loadtxt('output/MDMexact_cl.dat')
l_MDMexact = CMB_MDMexact[:,0]
Cl_TT_MDMexact = CMB_MDMexact[:,1]
Cl_EE_MDMexact = CMB_MDMexact[:,2]
Cl_TE_MDMexact = CMB_MDMexact[:,3]
#Ptheta
transfer_theta_tot_z0_MDMexact = np.loadtxt('output/MDMexact_z1_tk.dat', usecols=(12,))
transfer_theta_tot_z1_MDMexact = np.loadtxt('output/MDMexact_z2_tk.dat', usecols=(12,))
transfer_theta_tot_z2_MDMexact = np.loadtxt('output/MDMexact_z3_tk.dat', usecols=(12,))


interp_Pm_z0_MDMexact = log_interp1d(k_MDMexact,Pm_z0_MDMexact)
interp_Pm_z1_MDMexact = log_interp1d(k_MDMexact,Pm_z1_MDMexact)
interp_Pm_z2_MDMexact = log_interp1d(k_MDMexact,Pm_z2_MDMexact)
interp_Ptheta_z0_MDMexact = log_interp1d(k_MDMexact,transfer_theta_tot_z0_MDMexact**2.)
interp_Ptheta_z1_MDMexact = log_interp1d(k_MDMexact,transfer_theta_tot_z1_MDMexact**2.)
interp_Ptheta_z2_MDMexact = log_interp1d(k_MDMexact,transfer_theta_tot_z2_MDMexact**2.)
#Theta and delta @k=0.1Mpc^{-1}
tau_MDMexact=np.loadtxt('output/MDMexact_perturbations_k2_s.dat', usecols=(0,))
theta_mim_MDMexact=np.loadtxt('output/MDMexact_perturbations_k2_s.dat', usecols=(16,))
delta_mim_MDMexact=np.loadtxt('output/MDMexact_perturbations_k2_s.dat', usecols=(15,))
interp_theta_mim_MDMexact = log_interp1d(tau_MDMexact,theta_mim_MDMexact)
interp_delta_mim_MDMexact = log_interp1d(tau_MDMexact,abs(delta_mim_MDMexact))

###MDM50 quantities
#Pm
pk_z0_MDM50 = np.loadtxt('output/MDM50_z1_pk.dat')
pk_z1_MDM50 = np.loadtxt('output/MDM50_z2_pk.dat')
pk_z2_MDM50 = np.loadtxt('output/MDM50_z3_pk.dat')
k_MDM50 = pk_z0_MDM50[:,0]
Pm_z0_MDM50 = pk_z0_MDM50[:,1]
Pm_z1_MDM50 = pk_z1_MDM50[:,1]
Pm_z2_MDM50 = pk_z2_MDM50[:,1]

#CMB
CMB_MDM50 = np.loadtxt('output/MDM50_cl.dat')
l_MDM50 = CMB_MDM50[:,0]
Cl_TT_MDM50 = CMB_MDM50[:,1]
Cl_EE_MDM50 = CMB_MDM50[:,2]
Cl_TE_MDM50 = CMB_MDM50[:,3]
#Ptheta
transfer_theta_tot_z0_MDM50 = np.loadtxt('output/MDM50_z1_tk.dat', usecols=(12,))
transfer_theta_tot_z1_MDM50 = np.loadtxt('output/MDM50_z2_tk.dat', usecols=(12,))
transfer_theta_tot_z2_MDM50 = np.loadtxt('output/MDM50_z3_tk.dat', usecols=(12,))


interp_Pm_z0_MDM50 = log_interp1d(k_MDM50,Pm_z0_MDM50)
interp_Pm_z1_MDM50 = log_interp1d(k_MDM50,Pm_z1_MDM50)
interp_Pm_z2_MDM50 = log_interp1d(k_MDM50,Pm_z2_MDM50)
interp_Ptheta_z0_MDM50 = log_interp1d(k_MDM50,transfer_theta_tot_z0_MDM50**2.)
interp_Ptheta_z1_MDM50 = log_interp1d(k_MDM50,transfer_theta_tot_z1_MDM50**2.)
interp_Ptheta_z2_MDM50 = log_interp1d(k_MDM50,transfer_theta_tot_z2_MDM50**2.)
#Theta and delta @k=0.1Mpc^{-1}
tau_MDM50=np.loadtxt('output/MDM50_perturbations_k2_s.dat', usecols=(0,))
theta_mim_MDM50=np.loadtxt('output/MDM50_perturbations_k2_s.dat', usecols=(16,))
delta_mim_MDM50=np.loadtxt('output/MDM50_perturbations_k2_s.dat', usecols=(15,))
interp_theta_mim_MDM50 = log_interp1d(tau_MDM50,theta_mim_MDM50)
interp_delta_mim_MDM50 = log_interp1d(tau_MDM50,abs(delta_mim_MDM50))

###MDM20 quantities
#Pm
pk_z0_MDM20 = np.loadtxt('output/MDM20_z1_pk.dat')
pk_z1_MDM20 = np.loadtxt('output/MDM20_z2_pk.dat')
pk_z2_MDM20 = np.loadtxt('output/MDM20_z3_pk.dat')
k_MDM20 = pk_z0_MDM20[:,0]
Pm_z0_MDM20 = pk_z0_MDM20[:,1]
Pm_z1_MDM20 = pk_z1_MDM20[:,1]
Pm_z2_MDM20 = pk_z2_MDM20[:,1]

#CMB
CMB_MDM20 = np.loadtxt('output/MDM20_cl.dat')
l_MDM20 = CMB_MDM20[:,0]
Cl_TT_MDM20 = CMB_MDM20[:,1]
Cl_EE_MDM20 = CMB_MDM20[:,2]
Cl_TE_MDM20 = CMB_MDM20[:,3]
#Ptheta
transfer_theta_tot_z0_MDM20 = np.loadtxt('output/MDM20_z1_tk.dat', usecols=(12,))
transfer_theta_tot_z1_MDM20 = np.loadtxt('output/MDM20_z2_tk.dat', usecols=(12,))
transfer_theta_tot_z2_MDM20 = np.loadtxt('output/MDM20_z3_tk.dat', usecols=(12,))


interp_Pm_z0_MDM20 = log_interp1d(k_MDM20,Pm_z0_MDM20)
interp_Pm_z1_MDM20 = log_interp1d(k_MDM20,Pm_z1_MDM20)
interp_Pm_z2_MDM20 = log_interp1d(k_MDM20,Pm_z2_MDM20)
interp_Ptheta_z0_MDM20 = log_interp1d(k_MDM20,transfer_theta_tot_z0_MDM20**2.)
interp_Ptheta_z1_MDM20 = log_interp1d(k_MDM20,transfer_theta_tot_z1_MDM20**2.)
interp_Ptheta_z2_MDM20 = log_interp1d(k_MDM20,transfer_theta_tot_z2_MDM20**2.)
#Theta and delta @k=0.1Mpc^{-1}
tau_MDM20=np.loadtxt('output/MDM20_perturbations_k2_s.dat', usecols=(0,))
theta_mim_MDM20=np.loadtxt('output/MDM20_perturbations_k2_s.dat', usecols=(16,))
delta_mim_MDM20=np.loadtxt('output/MDM20_perturbations_k2_s.dat', usecols=(15,))
interp_theta_mim_MDM20 = log_interp1d(tau_MDM20,theta_mim_MDM20)
interp_delta_mim_MDM20 = log_interp1d(tau_MDM20,abs(delta_mim_MDM20))


###MDM10 quantities
#Pm
pk_z0_MDM10 = np.loadtxt('output/MDM10_z1_pk.dat')
pk_z1_MDM10 = np.loadtxt('output/MDM10_z2_pk.dat')
pk_z2_MDM10 = np.loadtxt('output/MDM10_z3_pk.dat')
k_MDM10 = pk_z0_MDM10[:,0]
Pm_z0_MDM10 = pk_z0_MDM10[:,1]
Pm_z1_MDM10 = pk_z1_MDM10[:,1]
Pm_z2_MDM10 = pk_z2_MDM10[:,1]

#CMB
CMB_MDM10 = np.loadtxt('output/MDM10_cl.dat')
l_MDM10 = CMB_MDM10[:,0]
Cl_TT_MDM10 = CMB_MDM10[:,1]
Cl_EE_MDM10 = CMB_MDM10[:,2]
Cl_TE_MDM10 = CMB_MDM10[:,3]
#Ptheta
transfer_theta_tot_z0_MDM10 = np.loadtxt('output/MDM10_z1_tk.dat', usecols=(12,))
transfer_theta_tot_z1_MDM10 = np.loadtxt('output/MDM10_z2_tk.dat', usecols=(12,))
transfer_theta_tot_z2_MDM10 = np.loadtxt('output/MDM10_z3_tk.dat', usecols=(12,))


interp_Pm_z0_MDM10 = log_interp1d(k_MDM10,Pm_z0_MDM10)
interp_Pm_z1_MDM10 = log_interp1d(k_MDM10,Pm_z1_MDM10)
interp_Pm_z2_MDM10 = log_interp1d(k_MDM10,Pm_z2_MDM10)
interp_Ptheta_z0_MDM10 = log_interp1d(k_MDM10,transfer_theta_tot_z0_MDM10**2.)
interp_Ptheta_z1_MDM10 = log_interp1d(k_MDM10,transfer_theta_tot_z1_MDM10**2.)
interp_Ptheta_z2_MDM10 = log_interp1d(k_MDM10,transfer_theta_tot_z2_MDM10**2.)
#Theta and delta @k=0.1Mpc^{-1}
tau_MDM10=np.loadtxt('output/MDM10_perturbations_k2_s.dat', usecols=(0,))
theta_mim_MDM10=np.loadtxt('output/MDM10_perturbations_k2_s.dat', usecols=(16,))
delta_mim_MDM10=np.loadtxt('output/MDM10_perturbations_k2_s.dat', usecols=(15,))
interp_theta_mim_MDM10 = log_interp1d(tau_MDM10,theta_mim_MDM10)
interp_delta_mim_MDM10 = log_interp1d(tau_MDM10,abs(delta_mim_MDM10))

#Plot Theta and Delta ratios
gs = gridspec.GridSpec(1, 2,top=0.9,bottom=0.15,left=0.1,right=0.95,wspace=.3)
fig = plt.figure(figsize=(14.0,5.0))
ax = plt.subplot(gs[0,0])
ax.semilogx(tau_GR,interp_theta_mim_MDMexact(tau_GR)/interp_theta_cdm_GR(tau_GR),lw=2,label=r'Adiabatic ICs')
ax.semilogx(tau_GR,interp_theta_mim_MDM50(tau_GR)/interp_theta_cdm_GR(tau_GR),lw=2,label=r'$\alpha=0.5$')
ax.semilogx(tau_GR,interp_theta_mim_MDM20(tau_GR)/interp_theta_cdm_GR(tau_GR),lw=2,label=r'$\alpha=0.2$')
ax.semilogx(tau_GR,interp_theta_mim_MDM10(tau_GR)/interp_theta_cdm_GR(tau_GR),lw=2,label=r'$\alpha=0.1$')
ax.set_ylabel(r'$\theta_{\rm MDM}/\theta_{\rm GR}$'
                 ,fontsize=20)
ax.set_xlabel(r'$\tau$',fontsize=20)
ax.tick_params(axis='both',width=1,length=8,labelsize=13)
ax.tick_params(axis='both',which='minor',width=1,length=3)
ax.legend(fontsize=15,ncol=1,loc=1)
ax.set_xlim(tau_GR[0],tau_GR[-1])

ax = plt.subplot(gs[0,1])
ax.semilogx(tau_GR,abs(interp_delta_mim_MDMexact(tau_GR)/interp_delta_cdm_GR(tau_GR)),lw=2,label=r'Adiabatic ICs')
ax.semilogx(tau_GR,abs(interp_delta_mim_MDM50(tau_GR)/interp_delta_cdm_GR(tau_GR)),lw=2,label=r'$\alpha=0.5$')
ax.semilogx(tau_GR,abs(interp_delta_mim_MDM20(tau_GR)/interp_delta_cdm_GR(tau_GR)),lw=2,label=r'$\alpha=0.2$')
ax.semilogx(tau_GR,abs(interp_delta_mim_MDM10(tau_GR)/interp_delta_cdm_GR(tau_GR)),lw=2,label=r'$\alpha=0.1$')
ax.set_ylabel(r'$\delta_{\rm MDM}/\delta_{\rm GR}$'
                 ,fontsize=20)
ax.set_xlabel(r'$\tau$',fontsize=20)
ax.tick_params(axis='both',width=1,length=8,labelsize=13)
ax.tick_params(axis='both',which='minor',width=1,length=3)
ax.set_xlim(tau_GR[0],tau_GR[-1])
plt.savefig('Plots/Final/Perts_ratios.pdf')


#cosmic variance ell
sigma_plus = 1+np.sqrt(2./(0.73*(2*l_GR+1)))
sigma_minus = 1-np.sqrt(2./(0.73*(2*l_GR+1)))

#Plot CMB ratios

gs = gridspec.GridSpec(1, 2,top=0.9,bottom=0.15,left=0.1,right=0.95,wspace=.3)
fig = plt.figure(figsize=(14.0,5.0))
ax = plt.subplot(gs[0,0])

ax.fill_between(l_GR,sigma_plus,sigma_minus,color='k',alpha=0.2)
ax.semilogx(l_GR,Cl_TT_MDMexact/Cl_TT_GR,lw=2,label=r'Adiabatic ICs')
ax.semilogx(l_GR,Cl_TT_MDM50/Cl_TT_GR,lw=2,label=r'$\alpha=0.5')
ax.semilogx(l_GR,Cl_TT_MDM20/Cl_TT_GR,lw=2,label=r'$\alpha=0.2$')
ax.semilogx(l_GR,Cl_TT_MDM10/Cl_TT_GR,lw=2,label=r'$\alpha=0.1$')
ax.set_ylabel(r'$\left(C_\ell^{\rm TT}\right)^{\rm MDM}/\left(C_\ell^{\rm TT}\right)^{\rm GR}$'
                 ,fontsize=20)
ax.set_xlabel(r'$\ell$',fontsize=20)
ax.tick_params(axis='both',width=1,length=8,labelsize=13)
ax.tick_params(axis='both',which='minor',width=1,length=3)
ax.legend(fontsize=15,ncol=1,loc=1)
ax.set_xlim(l_GR[0],l_GR[-1])


ax = plt.subplot(gs[0,1])
ax.fill_between(l_GR,sigma_plus,sigma_minus,color='k',alpha=0.2)
ax.semilogx(l_GR,Cl_EE_MDMexact/Cl_EE_GR,lw=2,label=r'Adiabatic ICs')
ax.semilogx(l_GR,Cl_EE_MDM50/Cl_EE_GR,lw=2,label=r'$\alpha=0.5$')
ax.semilogx(l_GR,Cl_EE_MDM20/Cl_EE_GR,lw=2,label=r'$\alpha=0.2$')
ax.semilogx(l_GR,Cl_EE_MDM10/Cl_EE_GR,lw=2,label=r'$\alpha=0.1$')
ax.set_ylabel(r'$\left(C_\ell^{\rm EE}\right)^{\rm MDM}/\left(C_\ell^{\rm EE}\right)^{\rm GR}$'
                 ,fontsize=20)
ax.set_xlabel(r'$\ell$',fontsize=20)
ax.tick_params(axis='both',width=1,length=8,labelsize=13)
ax.tick_params(axis='both',which='minor',width=1,length=3)
ax.set_xlim(l_GR[0],l_GR[-1])




plt.savefig('Plots/Final/CMB_ratios.pdf')




#Plot Pm ratios

#variance
n_g = 1.998e-3
V_s = 1.719e9
kint = np.logspace(np.log10(1e-3),0.2,65)
dk = np.diff(kint)
kint = 0.5*(kint[0:-1]+kint[1:])
sigma_m = np.sqrt(4*np.pi**2/(dk*V_s*kint**2))*(1+1/(n_g*interp_Pm_z1_GR(kint)))


#gs = gridspec.GridSpec(1, 1,top=0.9,bottom=0.15,left=0.1,right=0.95,wspace=.3)
fig = plt.figure(figsize=(12.0,8.0))
#ax = plt.subplot(gs[0,0])
plt.fill_between(kint,1+(sigma_m),1-(sigma_m),color='k',alpha=0.3)
plt.semilogx(kint,1+sigma_m,lw=3,color='m',label=r'variance')
plt.semilogx(kint,1-sigma_m,lw=3,color='m')
plt.semilogx(kint,interp_Pm_z1_MDMexact(kint)/interp_Pm_z1_GR(kint),lw=2,label=r'Adiabatic ICs')
plt.semilogx(kint,interp_Pm_z1_MDM50(kint)/interp_Pm_z1_GR(kint),lw=2,label=r'$\alpha=0.5$')
plt.semilogx(kint,interp_Pm_z1_MDM20(kint)/interp_Pm_z1_GR(kint),lw=2,label=r'$\alpha=0.2$')
plt.semilogx(kint,interp_Pm_z1_MDM10(kint)/interp_Pm_z1_GR(kint),lw=2,label=r'$\alpha=0.1$')
plt.ylabel(r'$\left(P_m\right)_{\rm MDM}/\left(P_m\right)_{\rm GR}$'
                 ,fontsize=20)
plt.xlabel(r'$k$ [$h$/Mpc] ',fontsize=20)
plt.tick_params(axis='both',width=1,length=8,labelsize=13)
plt.tick_params(axis='both',which='minor',width=1,length=3)
#plt.set_title(fontsize=15)
plt.legend(fontsize=18,ncol=1,loc=4)
plt.xlim(kint[0],kint[-1])
plt.ylim(0.7,1.4)


#plt.savefig('Plots/Final/Pm_ratios.pdf')
plt.savefig('Plots/Final/P_ratios.pdf')





plt.figure(figsize=(12,8))

ax1 = plt.subplot2grid((11, 14), (0, 0), rowspan=5, colspan=6)
ax2 = plt.subplot2grid((11, 14), (0, 8), rowspan=5, colspan=6)
ax3 = plt.subplot2grid((11, 14), (6, 4), rowspan=5, colspan=6)

#----------------------------------------------------------------------------------------------
# Upper Left Plot, Squeezed Shape
#----------------------------------------------------------------------------------------------
sigma_plus = 1+np.sqrt(2./(0.73*(2*l_GR+1)))
sigma_minus = 1-np.sqrt(2./(0.73*(2*l_GR+1)))
ax1.semilogx(l_GR,Cl_TT_MDMexact/Cl_TT_GR,
            color='#1f77b4', linestyle='-')
ax1.semilogx(l_GR,Cl_TT_MDM50/Cl_TT_GR,
            color='#ff7f0e', linestyle='-')
ax1.semilogx(l_GR,Cl_TT_MDM20/Cl_TT_GR,
            color='#2ca02c', linestyle='-')
ax1.semilogx(l_GR,Cl_TT_MDM10/Cl_TT_GR,
            color='#d62728', linestyle='-')
ax1.fill_between(l_GR,sigma_plus,sigma_minus,color='k',alpha=0.2)
ax1.set_xlim(l_GR[0],l_GR[-1])
ax1.set_ylim(np.amin(sigma_minus)-0.1,np.amax(Cl_TT_MDM50/Cl_TT_GR)+0.1)
ax1.set_xlabel(r'$\ell$', fontsize=15)
ax1.set_ylabel(r'$\left(C_\ell^{\rm TT}\right)^{\rm MDM}/\left(C_\ell^{\rm TT}\right)^{\rm GR}$'
                 ,fontsize=15)
ax1.tick_params(axis='both',width=1,length=8,labelsize=13)
ax1.tick_params(axis='both',which='minor',width=1,length=3)
#ax1.tick_params(axis='x', which='both', direction='in', bottom='on', 
#labelsize=15)
#ax1.tick_params(axis='x', which='major', width=0.9, length=7)
#ax1.tick_params(axis='x', which='minor', width=0.9, length=4)
#ax1.tick_params(axis='y', which='both', direction='in', right='on', 
#labelsize=15)
#ax1.tick_params(axis='y', which='major', width=0.8, length=6)
#ax1.tick_params(axis='y', which='minor', width=0.8, length=3)

#----------------------------------------------------------------------------------------------
# Center Plot, Equilateral Shape
#----------------------------------------------------------------------------------------------
ax2.semilogx(l_GR,Cl_EE_MDMexact/Cl_EE_GR,
            color='#1f77b4', linestyle='-')
ax2.semilogx(l_GR,Cl_EE_MDM50/Cl_EE_GR,
            color='#ff7f0e', linestyle='-')
ax2.semilogx(l_GR,Cl_EE_MDM20/Cl_EE_GR,
            color='#2ca02c', linestyle='-')
ax2.semilogx(l_GR,Cl_EE_MDM10/Cl_EE_GR,
            color='#d62728', linestyle='-')
ax2.fill_between(l_GR,sigma_plus,sigma_minus,color='k',alpha=0.2)
ax2.set_xlim(l_GR[0],l_GR[-1])
ax2.set_ylim(np.amin(sigma_minus)-0.1,np.amax(Cl_EE_MDM50/Cl_EE_GR)+0.1)
ax2.set_xlabel(r'$\ell$', fontsize=15)
ax2.set_ylabel(r'$\left(C_\ell^{\rm EE}\right)^{\rm MDM}/\left(C_\ell^{\rm EE}\right)^{\rm GR}$'
                 ,fontsize=15)
ax2.tick_params(axis='both',width=1,length=8,labelsize=13)
ax2.tick_params(axis='both',which='minor',width=1,length=3)
#ax2.tick_params(axis='x', which='both', direction='in', bottom='on', 
#labelsize=15)
#ax2.tick_params(axis='x', which='major', width=0.9, length=7)
#ax2.tick_params(axis='x', which='minor', width=0.9, length=4)
#ax2.tick_params(axis='y', which='both', direction='in', right='on', 
#labelsize=15)
#ax2.tick_params(axis='y', which='major', width=0.8, length=6)
#ax2.tick_params(axis='y', which='minor', width=0.8, length=3)

#----------------------------------------------------------------------------------------------
# Upper Right Plot, Folded Shape
#----------------------------------------------------------------------------------------------
n_g = 1.998e-3
V_s = 1.719e9
kint = np.logspace(np.log10(1e-3),np.log10(0.2),65)
dk = np.diff(kint)
kint = 0.5*(kint[0:-1]+kint[1:])
sigma_m = np.sqrt(4*np.pi**2/(np.log(np.amax(kint)/np.amin(kint))*16*V_s*kint**3))*(1+1/(n_g*interp_Pm_z1_GR(kint)))

ax3.semilogx(kint,interp_Pm_z1_MDMexact(kint)/interp_Pm_z1_GR(kint),
            color='#1f77b4', linestyle='-')
ax3.semilogx(kint,interp_Pm_z1_MDM50(kint)/interp_Pm_z1_GR(kint),
            color='#ff7f0e', linestyle='-')
ax3.semilogx(kint,interp_Pm_z1_MDM20(kint)/interp_Pm_z1_GR(kint),
            color='#2ca02c', linestyle='-')
ax3.semilogx(kint,interp_Pm_z1_MDM10(kint)/interp_Pm_z1_GR(kint),
            color='#d62728', linestyle='-')
ax3.fill_between(kint,1+(sigma_m),1-(sigma_m),color='k',alpha=0.3)
ax3.set_xlim(kint[0],kint[-1])
ax3.set_ylim(0.7,1.4)
#ax3.set_ylim(np.amin(interp_Pm_z1_MDM50(kint)/interp_Pm_z1_GR(kint)-0.01),np.amax(interp_Pm_z1_MDM50(kint)/interp_Pm_z1_GR(kint)+0.01))

ax3.set_xlabel(r'$k\ \left[h\mathrm{Mpc}^{-1}\right]$', fontsize=15)
ax3.set_ylabel(r'$\left(P_m\right)_{\rm MDM}/\left(P_m\right)_{\rm GR}$'
                 ,fontsize=15)
ax3.tick_params(axis='both',width=1,length=8,labelsize=13)
ax3.tick_params(axis='both',which='minor',width=1,length=3)
#ax3.tick_params(axis='x', which='both', direction='in', bottom='on', 
#labelsize=15)
#ax3.tick_params(axis='x', which='major', width=0.9, length=7)
#ax3.tick_params(axis='x', which='minor', width=0.9, length=4)
#ax3.tick_params(axis='y', which='both', direction='in', right='on', 
#labelsize=15)
#ax3.tick_params(axis='y', which='major', width=0.8, length=6)
#ax3.tick_params(axis='y', which='minor', width=0.8, length=3)

# 
# LEGEND
# 
Color_1 = mlines.Line2D([],[],color='#1f77b4')
Color_2 = mlines.Line2D([],[],color='#ff7f0e')
Color_3 = mlines.Line2D([],[],color='#2ca02c')
Color_4 = mlines.Line2D([],[],color='#d62728')
Color_5 = mlines.Line2D([],[],color='#e377c2')
Label_1 = r'Adiabatic ICs'
Label_2 = r'$\alpha=0.5$'
Label_3 = r'$\alpha=0.2$'
Label_4 = r'$\alpha=0.1$'


lgd = plt.figlegend((Color_1,Color_2,Color_3,Color_4),
                     (Label_1,Label_2,Label_3,Label_4),
                     loc='upper center', ncol=5, columnspacing=1, 
fontsize=15)

plt.savefig('Plots/Final/Spectra.pdf', bbox_inches='tight')
#close()
#show()
