import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset,RandomSampler
import time
import array
from pathlib import Path
import sys
import os
import matplotlib
from matplotlib import pyplot as plt

import mesh_gen
from bte_train import bte_train
from bte_test import bte_test

epochs = 30000
path = "./"
Nl = 5  # number of L samples
L = np.linspace(0, 4, Nl).reshape(-1,1)  # 0=10nm, 1=100nm, 2=1um, 3=10um, 4=100um ... 

Nx = 40  # number of spatial points
Nk = 10  # number of frequency bands
Ns = 16  # number of quadrature points
Np = 3   # number of phonon branches

a = 5.431  # lattice constant
Tr = 300   # reference temperature
dT = 0.5   # delta T

x,mu,w,k = mesh_gen.OneD_mesh(Nx,Ns,Nk,a)
vt0,vt1 = mesh_gen.OneD_vt(k,Tr)

#===============================================================
#=== model training
#===============================================================

learning_rate = 1e-3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

bte_train(x,mu,w,k,vt0,vt1,Nx,Ns,Nk,Np,L,Tr,dT,a,learning_rate,epochs,path,device)

#===============================================================
#=== model testing
#===============================================================

index = 1
Nl = 17
Nx = 40
Ns = 64

L = np.linspace(0, 4, Nl).reshape(-1,1)
x,mu,w,k = mesh_gen.OneD_mesh(Nx,Ns,Nk,a)

bte_test(x,mu,w,k,vt0,vt1,Nx,Ns,Nk,Np,L,Tr,dT,a,index,path,device)

#===============================================================
#=== results ploting
#===============================================================

Data = np.load(str(index)+'Kn_1d_ng.npz')
x = Data['x']
T = Data['T']
q = Data['q']
L = Data['L']-8

T1 = T[:,0].reshape(-1,1)
T2 = T[:,4].reshape(-1,1)
T3 = T[:,8].reshape(-1,1)
T4 = T[:,16].reshape(-1,1)

plt.figure(figsize=(6, 5))
plt.plot(x,T1,'r--',linewidth=2.5)
plt.plot(x,T2,'r--',linewidth=2.5)
plt.plot(x,T3,'r--',linewidth=2.5)
plt.plot(x,T4,'r--',linewidth=2.5)
plt.ylabel(r'$T^{*}$')
plt.xlabel(r'$X$')
plt.axis([0,1,0,1])
plt.savefig('T_1d.png', dpi=300, bbox_inches='tight')

plt.figure(figsize=(6, 5))
plt.plot(10**L[0:17:4],q[0:17:4],'ro',markeredgewidth=1.5,markersize=7)
plt.plot(10**L,q,'ro',fillstyle='none',markeredgewidth=1.5,markersize=7)
plt.xscale("log")
plt.ylabel(r'$k_{eff}/k_{bulk}$')
plt.xlabel(r'$L$ (m)')
plt.savefig('q_1d.png', dpi=300, bbox_inches='tight')


