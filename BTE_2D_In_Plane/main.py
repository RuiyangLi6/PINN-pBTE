import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset,RandomSampler
import time
import array
from pathlib import Path
import sys
import os
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple

from bte_train import bte_train
from bte_test import bte_test
from mesh_2d import *
from analytical import *

epochs = 4000
path = "./"
Nl = 5  # number of L samples
logL = np.linspace(0,4,Nl).reshape(-1,1)  # 0=10nm, 1=100nm, 2=1um, 3=10um, 4=100um ... 
batchsize = array.array('i', [100, 150])

############################################
Nx = 300 # number of spatial points
Nb = 45  # number of frequency bands
Nk = 10
N1 = N2 = 12
Np = 3   # number of phonon branches
Ns = N1*N2  # number of quadrature points

Tr = 300  # Reference temperature
dT = 0.5  # Temperature difference (DeltaT/2)

# Interior points
x,y,mu,eta,w,k = TwoD_train_mesh(Nx,N1,N2,Nk)
x = np.tile(x,(1,Nl)).reshape(-1,1)
y = np.tile(y,(1,Nl)).reshape(-1,1)
L = np.tile(logL,(Nx,1))

# Boundary points
xb = np.linspace(0,1,Nb+2)[1:Nb+1].reshape(-1,1)
xb,kb,Lb = np.meshgrid(xb,k,logL)
xb = xb.reshape(-1,1)
kb = kb.reshape(-1,1)
Lb = Lb.reshape(-1,1)

vt0,vt1 = TwoD_vt(k,Tr)
vt0b,vt1b = TwoD_vt(kb,Tr)
kb = kb/(np.pi*2/a)

#===============================================================
#=== model training
#===============================================================
learning_rate = 1e-3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

bte_train(x,y,mu,eta,L,w,k,vt0,vt1,xb,kb,vt0b,vt1b,Lb,Ns,Nk,Np,Tr,dT,batchsize,learning_rate,epochs,path,device)

#===============================================================
#=== model testing
#===============================================================
index = 1
Nl = 17
L = np.linspace(0,4,Nl).reshape(-1,1)
Nx = Ny = 40
N1 = N2 = 32
Ns = N1*N2

x,y,wy,mu,eta,w,k = TwoD_test_mesh(Nx,Ny,N1,N2,Nk)
vt0,vt1 = TwoD_vt(k,Tr)

bte_test(x,y,mu,eta,w,k,vt0,vt1,Nx,Ns,Nk,Np,L,Tr,dT,index,path,device)

#===============================================================
#=== results ploting
#===============================================================

Data = np.load(str(index)+'Kn_2d.npz')
T = Data['T']
q = Data['qx']
L = 10**(L-8) # unit m

y_unique = np.unique(y)
qxa = analytical_2d(y,L,Tr) # analytical solution to qx

qx = np.zeros((Ny,Nl))
for j in range(Nl):
    for i in range(Ny):
        qx[i,j] = np.mean(q[i*Nx:(i+1)*Nx,j])

k = np.sum(qx*wy,axis=0)
ka = np.sum(qxa*wy,axis=0)

index = np.array([6,8,12])
plt.figure(figsize=(6, 5))
plt.plot(y_unique,qxa[:,4],c='k',linewidth=3,label="Analytical")
plt.plot(y_unique,qx[:,4],'ro',fillstyle='none',markeredgewidth=1.5,markersize=7,label="PINN")
plt.plot(y_unique,qxa[:,index],c='k',linewidth=3)
plt.plot(y_unique,qx[:,index],'ro',fillstyle='none',markeredgewidth=1.5,markersize=7)
plt.ylabel(r'$q_{x}^{*}$')
plt.xlabel(r'$Y$')
plt.legend(frameon=False)
plt.xlim(0,1)
plt.savefig('qx_2d.png', dpi=300, bbox_inches='tight')


plt.figure(figsize=(6, 5))
p0, = plt.plot(L,ka,'k',linewidth=3)
p1, = plt.plot(L[0:Nl:4],k[0:Nl:4],'ro',markeredgewidth=1.5,markersize=7)
p2, = plt.plot(L,k,'ro',fillstyle='none',markeredgewidth=1.5,markersize=7)
plt.xscale("log")
plt.ylabel(r'$k_{eff}/k_{bulk}$')
plt.xlabel(r'$L$ (m)')
plt.legend([p0, (p1, p2)], ["Analytical","PINN"], numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)}, frameon=False)
plt.savefig('k_2d.png', dpi=300, bbox_inches='tight')


