import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset,RandomSampler
import time
from mesh_2d import *
from model import *

def bte_train(x,y,mu,eta,L,w,k,vt0,vt1,xb,yb,kb,vt0b,vt1b,Lb,Ns,Nk,Np,Nw,logL,Tr,dT,batchsize,learning_rate,epochs,path,device):
	dataset1 = TensorDataset(torch.Tensor(x),torch.Tensor(y),torch.Tensor(L))
	dataloader1 = DataLoader(dataset1,batch_size=batchsize[0],shuffle=True,num_workers=0,drop_last=True)
	dataset2 = TensorDataset(torch.Tensor(xb),torch.Tensor(kb),torch.Tensor(vt0b),torch.Tensor(vt1b),torch.Tensor(Lb))
	dataloader2 = DataLoader(dataset2,batch_size=batchsize[1],shuffle=True,num_workers=0,drop_last=True)

	################################################################
	net0 = Net(8, 8, 30).to(device)
	net1 = Net(3, 8, 30).to(device)

	optimizer0 = optim.Adam(net0.parameters(), lr=learning_rate, betas=(0.9,0.99), eps=1e-15)
	optimizer1 = optim.Adam(net1.parameters(), lr=learning_rate, betas=(0.9,0.99), eps=1e-15)

	def init_normal(m):
		if type(m) == nn.Linear:
			nn.init.kaiming_normal_(m.weight)

	net0.apply(init_normal)
	net0.train()
	net1.apply(init_normal)
	net1.train()

	############################################################################

	def criterion(x_in,y_in,L_in,xb,kb,vt0b,vt1b,Lb):
		TC = (1/3)*torch.sum(C*v**3*tau*wk)/(dT*4)*1e11/(10**L_in).to(device)
		x = x_in.repeat(1,Ns*Nk).reshape(-1,1).to(device)
		y = y_in.repeat(1,Ns*Nk).reshape(-1,1).to(device)
		L = L_in.repeat(1,Ns*Nk).reshape(-1,1).to(device)

		x.requires_grad = True
		y.requires_grad = True

		######### Interior points ##########
		e0_in = torch.cat((x,y,mu,eta,k,vt0,L,torch.zeros_like(x)),1).to(device)
		e1_in = torch.cat((x,y,mu,eta,k,vt1,L,torch.ones_like(x)),1).to(device)
		e0 = net0(e0_in)*dT*(10**vt0)/(10**L)
		e1 = net0(e1_in)*dT*(10**vt1)/(10**L)
		eEq = net1(torch.cat((x,y,L),1))*dT

		e0_x = torch.autograd.grad(e0+eEq,x,grad_outputs=torch.ones_like(x).to(device),create_graph=True)[0]
		e1_x = torch.autograd.grad(e1+eEq,x,grad_outputs=torch.ones_like(x).to(device),create_graph=True)[0]
		e0_y = torch.autograd.grad(e0+eEq,y,grad_outputs=torch.ones_like(y).to(device),create_graph=True)[0]
		e1_y = torch.autograd.grad(e1+eEq,y,grad_outputs=torch.ones_like(y).to(device),create_graph=True)[0]

		e = torch.cat(((e0+eEq).reshape(-1,Ns*Nk),(e0+eEq).reshape(-1,Ns*Nk),(e1+eEq).reshape(-1,Ns*Nk)),1).reshape(-1,1)
		e_x = torch.cat((e0_x.reshape(-1,Ns*Nk),e0_x.reshape(-1,Ns*Nk),e1_x.reshape(-1,Ns*Nk)),1).reshape(-1,1)
		e_y = torch.cat((e0_y.reshape(-1,Ns*Nk),e0_y.reshape(-1,Ns*Nk),e1_y.reshape(-1,Ns*Nk)),1).reshape(-1,1)
		sum_e = torch.matmul(e.reshape(-1,Ns), w).reshape(-1,1).to(device)
		deltaT = torch.matmul(sum_e.reshape(-1,Nk*Np),C*wk/tau*v/(4*np.pi)).reshape(-1,1).repeat(1,Nk*Ns).reshape(-1,1)/torch.sum(C/tau*wk*v)

		sum_ex = torch.matmul(e_x.reshape(-1,Ns), w*mu[0:Ns].reshape(-1,1)).reshape(-1,1)
		sum_ey = torch.matmul(e_y.reshape(-1,Ns), w*eta[0:Ns].reshape(-1,1)).reshape(-1,1)
		dq = torch.matmul((sum_ex+sum_ey).reshape(-1,Nk*Np),C*wk*v**2/(4*np.pi)).reshape(-1,1)

		######### Isothermal boundary ##########
		xb = xb.repeat(1,int(Ns/2)).reshape(-1,1).to(device)
		kb = kb.repeat(1,int(Ns/2)).reshape(-1,1).to(device)
		Lb = Lb.repeat(1,int(Ns/2)).reshape(-1,1).to(device)
		vt0b = vt0b.repeat(1,int(Ns/2)).reshape(-1,1).to(device)
		vt1b = vt1b.repeat(1,int(Ns/2)).reshape(-1,1).to(device)
		pc = torch.cat((torch.zeros(Nb*3,1),torch.ones(Nb*3,1)),0).to(device)
		ph = torch.cat((torch.zeros(Nb,1),torch.ones(Nb,1)),0).to(device)

		x1 = torch.cat((torch.ones_like(xb),xb*0.9),1)
		x0 = torch.cat((torch.zeros_like(xb),xb*0.9),1)
		y1 = torch.cat((xb*0.8+0.1,torch.ones_like(xb)),1)
		y0 = torch.cat((xb,torch.zeros_like(xb)),1)

		cb = torch.cat((y0,x0,x1),0).repeat(2,1)
		cEq = net1(torch.cat((cb,Lb.repeat(6,1)),1))
		vtc = torch.cat((vt0b.repeat(3,1),vt1b.repeat(3,1)),0) # v*tau for cold boundary
		c_in = torch.cat((cb,sb,kb.repeat(6,1),vtc,Lb.repeat(6,1),pc),1)
		ec1 = net0(c_in)*(10**vtc)/(10**Lb.repeat(6,1)) + cEq

		hEq = net1(torch.cat((y1.repeat(2,1),Lb.repeat(2,1)),1))
		vth = torch.cat((vt0b,vt1b),0)
		h_in = torch.cat((y1.repeat(2,1),sy1,kb.repeat(2,1),vth,Lb.repeat(2,1),ph),1)
		eh1 = net0(h_in)*(10**vth)/(10**Lb.repeat(2,1)) + hEq

		# output of corner points
		ec2 = net0(c0_in)*(10**vtw)/(10**Lw) + net1(c1_in)
		eh2 = net0(h0_in)*(10**vtw)/(10**Lw) + net1(h1_in)

		######### Loss ##########
		loss_1 = ((mu*e0_x+eta*e0_y) + e0/(10**vt0/10**L))/dT
		loss_2 = ((mu*e1_x+eta*e1_y) + e1/(10**vt1/10**L))/dT
		loss_3 = (deltaT - eEq)/dT
		loss_4 = (dq/TC)
		loss_5 = (ec1 + 1)
		loss_6 = (ec2 + 1)
		loss_7 = (eh1 - 1)
		loss_8 = (eh2 - 1)

		##############
		# MSE LOSS
		loss_f = nn.MSELoss()

		loss1 = loss_f(loss_1,torch.zeros_like(loss_1))
		loss2 = loss_f(loss_2,torch.zeros_like(loss_2))
		loss3 = loss_f(loss_3,torch.zeros_like(loss_3))
		loss4 = loss_f(loss_4,torch.zeros_like(loss_4))
		loss5 = loss_f(loss_5,torch.zeros_like(loss_5))
		loss6 = loss_f(loss_6,torch.zeros_like(loss_6))
		loss7 = loss_f(loss_7,torch.zeros_like(loss_7))
		loss8 = loss_f(loss_8,torch.zeros_like(loss_8))

		return loss1, loss2, loss3, loss4, loss5, loss6, loss7, loss8

	###################################################################

	# Main loop
	Loss_min = 100
	Loss_list = []
	tic = time.time()

	wk = np.pi*2/a/Nk
	p = np.vstack((np.zeros((Nk,1)),np.zeros((Nk,1)),np.ones((Nk,1))))
	v,tau,C = param(np.tile(k,(Np,1)),p,Tr)
	v = torch.FloatTensor(v).to(device)
	tau = torch.FloatTensor(tau).to(device)
	C = torch.FloatTensor(C).to(device)

	Nb = int(Ns/2)*batchsize[1]
	s = np.hstack((mu,eta))
	mu = torch.FloatTensor(mu).repeat(batchsize[0]*Nk,1).to(device)
	eta = torch.FloatTensor(eta).repeat(batchsize[0]*Nk,1).to(device)
	w = torch.FloatTensor(w).to(device)

	# Solid angles for the points at the boundary
	sx0 = np.tile(s[s[:,0]>0],(batchsize[1],1))
	sx1 = np.tile(s[s[:,0]<0],(batchsize[1],1))
	sy0 = np.tile(s[s[:,1]>0],(batchsize[1],1))
	sy1 = np.tile(s[s[:,1]<0],(batchsize[1],1))
	sb = torch.FloatTensor(np.vstack((sy0,sx0,sx1))).repeat(2,1).to(device) # for cold boundary
	sy1 = torch.FloatTensor(sy1).repeat(2,1).to(device) # for hot top boundary

	k = k/(np.pi*2/a)
	Nl = len(logL)
	# extra corner points at the cold boundary (0.9 < Y < 1, X = 0 or X = 1)
	# we need these addtional training points as the boundary temperature distribution is discontinuous near the top corner
	yb = torch.FloatTensor(yb).repeat(1,int(Ns/2)*Nk*Nl).reshape(-1,1)
	xc0 = torch.cat((torch.zeros_like(yb),yb*0.1+0.9),1)
	xc1 = torch.cat((torch.ones_like(yb),yb*0.1+0.9),1)
	sc0 = np.tile(s[s[:,0]>0],(Nk*Nl*Nw,1))
	sc1 = np.tile(s[s[:,0]<0],(Nk*Nl*Nw,1))
	sc = torch.FloatTensor(np.concatenate((sc0,sc1),0)).repeat(2,1).to(device)
	xc = torch.cat((xc0,xc1),0).repeat(2,1).to(device)

	# extra corner points at the hot top boundary (0 < X < 0.1 or 0.9 < X < 1, Y = 1)
	yh0 = torch.cat((yb*0.1,torch.ones_like(yb)),1)
	yh1 = torch.cat((yb*0.1+0.9,torch.ones_like(yb)),1)
	sh = np.tile(s[s[:,1]<0],(Nk*Nl*Nw,1))
	sh = torch.FloatTensor(sh).repeat(4,1).to(device)
	yh = torch.cat((yh0,yh1),0).repeat(2,1).to(device)

	# phonon quantities for corner points (v*tau, wave number, and polarization)
	vt0w = torch.FloatTensor(vt0).repeat(1,int(Ns/2)).reshape(-1,1).repeat(2*Nw*Nl,1)
	vt1w = torch.FloatTensor(vt1).repeat(1,int(Ns/2)).reshape(-1,1).repeat(2*Nw*Nl,1)
	kw = torch.FloatTensor(k).repeat(1,int(Ns/2)).reshape(-1,1).repeat(4*Nw*Nl,1).to(device)
	vtw = torch.cat((vt0w,vt1w),0).to(device)
	pw = torch.cat((torch.zeros_like(vt0w),torch.ones_like(vt1w)),0).to(device)
	Lw = torch.FloatTensor(logL).repeat(1,int(Ns/2)*Nk).reshape(-1,1).repeat(Nw*4,1).to(device)

	# input to the network for these corner points
	c0_in = torch.cat((xc,sc,kw,vtw,Lw,pw),1)
	c1_in = torch.cat((xc,Lw),1)
	h0_in = torch.cat((yh,sh,kw,vtw,Lw,pw),1)
	h1_in = torch.cat((yh,Lw),1)

	vt0 = torch.FloatTensor(vt0).repeat(1,Ns).reshape(-1,1).repeat(batchsize[0],1).to(device)
	vt1 = torch.FloatTensor(vt1).repeat(1,Ns).reshape(-1,1).repeat(batchsize[0],1).to(device)
	k = torch.FloatTensor(k).repeat(1,Ns).reshape(-1,1).repeat(batchsize[0],1).to(device)

	for epoch in range(epochs):
		Loss = []
		for batch_idx, ((x_in,y_in,L_in),(xb,kb,vt0b,vt1b,Lb)) in enumerate(zip(dataloader1,dataloader2)):
			net0.zero_grad()
			net1.zero_grad()
			loss1,loss2,loss3,loss4,loss5,loss6,loss7,loss8 = criterion(x_in,y_in,L_in,xb,kb,vt0b,vt1b,Lb)
			loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7 + loss8
			loss.backward()
			optimizer0.step() 
			optimizer1.step() 
			Loss.append(loss.item())
			Loss_list.append([loss1.item(),loss2.item(),loss3.item(),loss4.item(),loss5.item(),loss6.item(),loss7.item(),loss8.item()])
			if epoch%200 == 0:
				print('Train Epoch: {}  Loss: {:.4f}  {:.4f}  {:.4f}  {:.4f}  {:.4f}'.format(epoch,loss1.item(),loss5.item(),loss6.item(),loss7.item(),loss8.item()))
				torch.save(net0.state_dict(),path+"train_ng_epoch"+str(epoch)+"e.pt")
		Loss = np.array(Loss)
		if np.mean(Loss) < Loss_min:
			torch.save(net0.state_dict(),path+"model0.pt")
			torch.save(net1.state_dict(),path+"model1.pt")
			Loss_min = np.mean(Loss)

	toc = time.time()
	elapseTime = toc - tic
	print("elapse time in parallel = ", elapseTime)
	np.savetxt(path+'Loss.txt',np.array(Loss_list), fmt='%.6f')

def bte_test(x,y,mu,eta,w,k,vt0,vt1,Nx,Ns,Nk,Np,L,Tr,dT,index,path,device):
	net0 = Net(8, 8, 30).to(device)
	net1 = Net(3, 8, 30).to(device)

	net0.load_state_dict(torch.load(path+"model0.pt",map_location=device))
	net0.eval()

	net1.load_state_dict(torch.load(path+"model1.pt",map_location=device))
	net1.eval()

	########################################
	p = np.vstack((np.zeros((Nk,1)),np.zeros((Nk,1)),np.ones((Nk,1))))
	v,tau,C = param(np.tile(k,(Np,1)),p,Tr)
	v = torch.FloatTensor(v).to(device)
	tau = torch.FloatTensor(tau).to(device)
	C = torch.FloatTensor(C).to(device)

	mu = torch.FloatTensor(mu).repeat(Nx*Nk,1).to(device)
	eta = torch.FloatTensor(eta).repeat(Nx*Nk,1).to(device)
	k = torch.FloatTensor(k/(np.pi*2/a)).repeat(1,Ns).reshape(-1,1).repeat(Nx,1).to(device)
	vt0 = torch.FloatTensor(vt0).repeat(1,Ns).reshape(-1,1).repeat(Nx,1).to(device)
	vt1 = torch.FloatTensor(vt1).repeat(1,Ns).reshape(-1,1).repeat(Nx,1).to(device)
	w = torch.FloatTensor(w).to(device)
	wk = np.pi*2/a/Nk

	deltaT = np.zeros((Nx**2,len(L)))
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
			deltaT[i*Nx:(i+1)*Nx,j] = np.squeeze(T.cpu().data.numpy())

	np.savez(str(int(index))+'Square',x = x,y = y,T = (deltaT+dT)/(2*dT),L = L)
	toc = time.time()
	elapseTime = toc - tic
	print ("elapse time = ", elapseTime)

