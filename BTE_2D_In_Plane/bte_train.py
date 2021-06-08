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

def bte_train(x,y,mu,eta,L,w,k,vt0,vt1,xb,kb,vt0b,vt1b,Lb,Ns,Nk,Np,Tr,dT,batchsize,learning_rate,epochs,path,device):
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
		TC = (1/3)*torch.sum(C*v**3*tau*wk)/(dT*2)*1e11/(10**L_in).to(device)
		x = x_in.repeat(1,Ns*Nk).reshape(-1,1).to(device)
		y = y_in.repeat(1,Ns*Nk).reshape(-1,1).to(device)
		L = L_in.repeat(1,Ns*Nk).reshape(-1,1).to(device)

		x.requires_grad = True
		y.requires_grad = True
		mu.requires_grad = True
		eta.requires_grad = True
		k.requires_grad = True
		L.requires_grad = True
		vt0.requires_grad = True
		vt1.requires_grad = True

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
		dq = torch.matmul((sum_ex+sum_ey).reshape(-1,Nk*Np), C*wk*(v**2)/(4*np.pi)).reshape(-1,1)

		######### Diffusely reflected boundary (y = 0 and y = 1) ##########
		xd = xb.repeat(1,int(Ns/2)).reshape(-1,1).repeat(2,1).to(device)
		kx = kb.repeat(1,int(Ns/2)).reshape(-1,1).repeat(2,1).to(device)
		Ld = Lb.repeat(1,int(Ns/2)).reshape(-1,1).repeat(2,1).to(device)
		vt0x = vt0b.repeat(1,int(Ns/2)).reshape(-1,1).to(device)
		vt1x = vt1b.repeat(1,int(Ns/2)).reshape(-1,1).to(device)
		vtx = torch.cat((vt0x,vt1x),0).to(device)
		px = torch.cat((torch.zeros(int(Npb/2),1),torch.ones(int(Npb/2),1)),0).to(device)

		dtEq = net1(torch.cat((xd,torch.ones_like(xd),Ld),1))
		dbEq = net1(torch.cat((xd,torch.zeros_like(xd),Ld),1))

		d_in = torch.cat((xd,torch.ones_like(xd),s1.repeat(batchsize[1]*2,1),kx,vtx,Ld,px),1).to(device)
		dt1 = net0(d_in)*(10**vtx)/(10**Ld) + dtEq

		d_in = torch.cat((xd,torch.ones_like(xd),s2.repeat(batchsize[1]*2,1),kx,vtx,Ld,px),1).to(device)
		dt2 = net0(d_in)*(10**vtx)/(10**Ld) + dtEq

		d_in = torch.cat((xd,torch.zeros_like(xd),s2.repeat(batchsize[1]*2,1),kx,vtx,Ld,px),1).to(device)
		db1 = net0(d_in)*(10**vtx)/(10**Ld) + dbEq

		d_in = torch.cat((xd,torch.zeros_like(xd),s1.repeat(batchsize[1]*2,1),kx,vtx,Ld,px),1).to(device)
		db2 = net0(d_in)*(10**vtx)/(10**Ld) + dbEq

		dt2 = 1/scale*torch.matmul(dt2.reshape(-1,int(Ns/2)),w2*s2[:,1].reshape(-1,1)).reshape(-1,1).repeat(1,int(Ns/2)).reshape(-1,1)
		db2 = 1/scale*torch.matmul(db2.reshape(-1,int(Ns/2)),(-w1)*s1[:,1].reshape(-1,1)).reshape(-1,1).repeat(1,int(Ns/2)).reshape(-1,1)

		######### Periodic boundary (x = 0 and x = 1) ##########
		yp = xb.repeat(1,Ns).reshape(-1,1).repeat(2,1).to(device)
		ky = kb.repeat(1,Ns).reshape(-1,1).repeat(2,1).to(device)
		Lp = Lb.repeat(1,Ns).reshape(-1,1).repeat(2,1).to(device)
		vt0y = vt0b.repeat(1,Ns).reshape(-1,1).to(device)
		vt1y = vt1b.repeat(1,Ns).reshape(-1,1).to(device)
		vty = torch.cat((vt0y,vt1y),0).to(device)
		py = torch.cat((torch.zeros(Npb,1),torch.ones(Npb,1)),0).to(device)
		no_x = torch.cat((yp,mu[0:Npb*2].reshape(-1,1),eta[0:Npb*2].reshape(-1,1),ky,vty,Lp,py),1)

		rEq = net1(torch.cat((torch.ones_like(yp),yp,Lp),1))
		lEq = net1(torch.cat((torch.zeros_like(yp),yp,Lp),1))

		l_in = torch.cat((torch.zeros_like(yp),no_x),1).to(device)
		r_in = torch.cat((torch.ones_like(yp),no_x),1).to(device)
		er = net0(r_in)*(10**vty)/(10**Lp) + rEq
		el = net0(l_in)*(10**vty)/(10**Lp) + lEq

		######### Loss ##########
		loss_1 = ((mu*e0_x+eta*e0_y) + e0/(10**vt0/10**L))/dT
		loss_2 = ((mu*e1_x+eta*e1_y) + e1/(10**vt1/10**L))/dT
		loss_3 = (deltaT - eEq)/dT
		loss_4 = (dq/TC)
		loss_5 = (dt1 - dt2)
		loss_6 = (db1 - db2)
		loss_7 = (el - er - 2)

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

		return loss1, loss2, loss3, loss4, loss5, loss6, loss7

	###################################################################

	# Main loop
	Loss_min = 100
	Loss_list = []
	tic = time.time()

	wk = np.pi*2/a/Nk
	p = np.concatenate((np.zeros((Nk,1)),np.zeros((Nk,1)),np.ones((Nk,1))),0)
	v,tau,C = param(np.tile(k,(Np,1)),p,Tr)
	v = torch.FloatTensor(v).to(device)
	tau = torch.FloatTensor(tau).to(device)
	C = torch.FloatTensor(C).to(device)

	s1,s2,w1,w2,scale = Boundary_diffuse(mu,eta,w)
	s1 = torch.FloatTensor(s1).to(device)
	s2 = torch.FloatTensor(s2).to(device)
	w1 = torch.FloatTensor(w1).to(device)
	w2 = torch.FloatTensor(w2).to(device)

	Npb = Ns*batchsize[1]
	mu = torch.FloatTensor(mu).repeat(batchsize[0]*Nk,1).to(device)
	eta = torch.FloatTensor(eta).repeat(batchsize[0]*Nk,1).to(device)
	w = torch.FloatTensor(w).to(device)

	vt0 = torch.FloatTensor(vt0).repeat(1,Ns).reshape(-1,1).repeat(batchsize[0],1).to(device)
	vt1 = torch.FloatTensor(vt1).repeat(1,Ns).reshape(-1,1).repeat(batchsize[0],1).to(device)

	k = k/(np.pi*2/a)
	k = torch.FloatTensor(k).repeat(1,Ns).reshape(-1,1).repeat(batchsize[0],1).to(device)

	for epoch in range(epochs):
		Loss = []
		for batch_idx, ((x_in,y_in,L_in),(xb,kb,vt0b,vt1b,Lb)) in enumerate(zip(dataloader1,dataloader2)):
			net0.zero_grad()
			net1.zero_grad()
			loss1,loss2,loss3,loss4,loss5,loss6,loss7 = criterion(x_in,y_in,L_in,xb,kb,vt0b,vt1b,Lb)
			loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7
			loss.backward()
			optimizer0.step() 
			optimizer1.step() 
			Loss.append(loss.item())
			Loss_list.append([loss1.item(),loss2.item(),loss3.item(),loss4.item(),loss5.item(),loss6.item(),loss7.item()])
			if epoch%200 == 0:
				print('Train Epoch: {}  Loss: {:.4f}  {:.4f}  {:.4f}  {:.4f}  {:.4f}'.format(epoch,loss1.item(),loss3.item(),loss5.item(),loss6.item(),loss7.item()))
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


