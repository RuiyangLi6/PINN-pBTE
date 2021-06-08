from mesh_2d import *
import numpy as np

# Analytical solution to 2D in-plane heat transfer
def analytical_2d(y,L,Tr):
    Np = 3
    Nk = N2 = 100

    y_unique = np.reshape(np.unique(y),(-1,1))
    Ny = len(y_unique)
    Nl = len(L)

    eta,w = np.polynomial.legendre.leggauss(N2)
    eta = (eta+1)/2
    w = w/2
    k, wk = np.polynomial.legendre.leggauss(Nk)
    k = np.reshape((k+1)*np.pi/a,(-1,1))
    wk = np.reshape(wk,(-1,1))*np.pi/a
    wk = np.tile(wk,(Np,1))
    k = np.tile(k,(Np,1))
    p = np.concatenate((np.zeros((Nk,1)),np.zeros((Nk,1)),np.ones((Nk,1))),0)

    v,tau,C = param(k,p,Tr)
    TC = (1/3)*np.sum(C*v**3*tau*wk)
    qx = np.zeros((Ny,Nl))
    temp = np.zeros((Np*Nk,1))

    for n in range(Nl):
        Kn = v*tau*1e13/(L[n]*1e10)
        for i in range(Ny):
            for j in range(Np*Nk):
                temp[j] = np.sum(w*(1-eta**2)*(2-np.exp(-y_unique[i]/(eta*Kn[j]))-np.exp((y_unique[i]-1)/(eta*Kn[j]))))
            qx[i,n] = np.sum(C*v**3*tau*wk*temp)

    qx = qx/(4*TC)

    return qx
