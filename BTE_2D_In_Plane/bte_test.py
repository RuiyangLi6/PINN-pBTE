import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from model import *
from mesh_2d import *

def bte_test(x,y,mu,eta,w,k,vt0,vt1,Nx,Ns,Nk,Np,L,Tr,dT,index,path,device):
	net0 = Net(8, 8, 30).to(device)
	net0.load_state_dict(torch.load(path+"model0.pt",map_location=device))
	net0.eval()

	net1 = Net(3, 8, 30).to(device)
	net1.load_state_dict(torch.load(path+"model1.pt",map_location=device))
	net1.eval()

	########################################
	p = np.concatenate((np.zeros((Nk,1)),np.zeros((Nk,1)),np.ones((Nk,1))),0)
	v,tau,C = param(np.tile(k,(Np,1)),p,Tr)
	v = torch.FloatTensor(v).to(device)
	tau = torch.FloatTensor(tau).to(device)
	C = torch.FloatTensor(C).to(device)

	mu = torch.FloatTensor(mu).repeat(Nx*Nk,1).to(device)
	eta = torch.FloatTensor(eta).repeat(Nx*Nk,1).to(device)
	k = torch.FloatTensor(k/(np.pi*2/5.431)).repeat(1,Ns).reshape(-1,1).repeat(Nx,1).to(device)
	vt0 = torch.FloatTensor(vt0).repeat(1,Ns).reshape(-1,1).repeat(Nx,1).to(device)
	vt1 = torch.FloatTensor(vt1).repeat(1,Ns).reshape(-1,1).repeat(Nx,1).to(device)
	w = torch.FloatTensor(w).to(device)
	wk = np.pi*2/5.431/Nk
	TC = (1/3)*torch.sum(C*v**3*tau*wk)*(dT*2)*1e11

	deltaT = np.zeros((Nx**2,len(L)))
	qx = np.zeros((Nx**2,len(L)))
	tic = time.time()
	for j in range(len(L)):
		for i in range(Nx):
			x1 = torch.FloatTensor(x[i*Nx:(i+1)*Nx]).repeat(1,Ns*Nk).reshape(-1,1).to(device)
			y1 = torch.FloatTensor(y[i*Nx:(i+1)*Nx]).repeat(1,Ns*Nk).reshape(-1,1).to(device)
			L1 = torch.FloatTensor(L[j]).repeat(Ns*Nk*Nx,1).to(device)

			eEq = net1(torch.cat((x1,y1,L1),1))*dT
			e0_in = torch.cat((x1,y1,mu,eta,k,vt0,L1,torch.zeros_like(x1)),1)
			e1_in = torch.cat((x1,y1,mu,eta,k,vt1,L1,torch.ones_like(x1)),1)
			e0 = net0(e0_in)*dT*(10**vt0)/(10**L1) + eEq
			e1 = net0(e1_in)*dT*(10**vt1)/(10**L1) + eEq
			e = torch.cat((e0.reshape(-1,Ns*Nk),e0.reshape(-1,Ns*Nk),e1.reshape(-1,Ns*Nk)),1).reshape(-1,1)

			sum_e = torch.matmul(e.reshape(-1,Ns), w).reshape(-1,1)
			T = torch.matmul(sum_e.reshape(-1,Nk*Np),C*wk/tau*v/(4*np.pi)).reshape(-1,1)/torch.sum(C/tau*wk*v)
			sum_ex = torch.matmul(e.reshape(-1,Ns), w*mu[0:Ns].reshape(-1,1)).reshape(-1,1)

			qx[i*Nx:(i+1)*Nx,j] = np.squeeze(torch.matmul(sum_ex.reshape(-1,Nk*Np), C*wk*v**2/(4*np.pi)).reshape(-1,1).cpu().data.numpy()/TC.cpu().data.numpy()*(10**L[j]))
			deltaT[i*Nx:(i+1)*Nx,j] = np.squeeze(T.cpu().data.numpy())

	np.savez(str(int(index))+'Kn_2d',x = x,y = y,T = (deltaT+dT)/(2*dT),qx = qx,L = L)
	toc = time.time()
	elapseTime = toc - tic
	print ("elapse time = ", elapseTime)


