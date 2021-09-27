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
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from bte_train import bte_train, bte_test
from mesh_2d import *

# Please note that the current model is not perfect for such a problem with harsh boundary conditions (step function at the top corners)
# Training would be better if the temperature distributions are continuous at the boundaries
epochs = 5000
path = "./"
Nl = 4
Nw = 10 # Number of points near the top corner
logL = np.linspace(0, 3, Nl).reshape(-1,1)
batchsize = array.array('i', [120, 80])
batchnum = 15

############################################
Nx = 450
Nb = 30
Nk = 10
N1 = N2 = 12 # number of quadrature points
Np = 3
Ns = N1*N2

Tr = 300
dT = 0.5

x,y,mu,eta,w,k = TwoD_train_mesh(Nx,200,N1,N2,Nk) # nonuniform spatial mesh
x = np.tile(x,(1,Nl)).reshape(-1,1)
y = np.tile(y,(1,Nl)).reshape(-1,1)
L = np.tile(logL,(Nx,1))

# Since the boundary condition is discontinuous near the top corner (i.e., Th = 1, Tc = -1),
# we have divided the boudnary points into two parts, one portion is near the top corner, the other is away from the top corner
xb = np.linspace(0,1,Nb+2)[1:Nb+1].reshape(-1,1)
yb = np.linspace(0,1,Nw+2)[1:Nw+1].reshape(-1,1)

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

bte_train(x,y,mu,eta,L,w,k,vt0,vt1,xb,yb,kb,vt0b,vt1b,Lb,Ns,Nk,Np,Nw,logL,Tr,dT,batchsize,learning_rate,epochs,path,device)

#===============================================================
#=== model testing
#===============================================================

index = 1
Nl = 4
logL = np.linspace(0,3,Nl).reshape(-1,1)
Nx = Ny = 51
N1 = N2 = 24

x,y,mu,eta,w,k = TwoD_test_mesh(Nx,N1,N2,Nk)

bte_test(x,y,mu,eta,w,k,vt0,vt1,Nx,N1*N2,Nk,Np,logL,Tr,dT,index,path,device)

#===============================================================
#=== results ploting
#===============================================================

oldcmp = cm.get_cmap('rainbow', 512)
newcmp = ListedColormap(oldcmp(np.linspace(0.1, 1, 256)))

Data = np.load(str(index)+'Square.npz')
x = Data['x']
y = Data['y']
T = Data['T']
T[T < 0] = 0
        
T0 = T[:,0].reshape(-1,1)
T1 = T[:,1].reshape(-1,1)
T2 = T[:,2].reshape(-1,1)
T3 = T[:,3].reshape(-1,1)

fig, axs = plt.subplots(2, 2, figsize=(12, 10))
im = axs[0, 0].contourf(x.reshape(Nx,Nx),y.reshape(Nx,Nx),T0.reshape(Nx,Nx),cmap=newcmp,levels=np.linspace(0,1,11))
im = axs[0, 1].contourf(x.reshape(Nx,Nx),y.reshape(Nx,Nx),T2.reshape(Nx,Nx),cmap=newcmp,levels=np.linspace(0,1,11))
im = axs[1, 0].contourf(x.reshape(Nx,Nx),y.reshape(Nx,Nx),T1.reshape(Nx,Nx),cmap=newcmp,levels=np.linspace(0,1,11))
im = axs[1, 1].contourf(x.reshape(Nx,Nx),y.reshape(Nx,Nx),T3.reshape(Nx,Nx),cmap=newcmp,levels=np.linspace(0,1,11))
for ax in axs.flat:
    ax.set_aspect('equal', adjustable='box')
    ax.set(xlabel=r'$X$', ylabel=r'$Y$')
fig.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.2, wspace=0.3)
fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.6)
plt.savefig('T_square.png', dpi=400, bbox_inches='tight')

