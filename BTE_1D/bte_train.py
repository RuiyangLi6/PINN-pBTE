import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset,RandomSampler
import time
from mesh_gen import param
import model

def bte_train(x,mu,w,k,vt0,vt1,Nx,Ns,Nk,Np,L,Tr,dT,a,learning_rate,epochs,path,device):
	net0 = model.Net(6, 8, 30).to(device)
	net1 = model.Net(2, 8, 30).to(device)

	optimizer0 = optim.Adam(net0.parameters(), lr=learning_rate, betas=(0.9,0.99), eps=1e-10)
	optimizer1 = optim.Adam(net1.parameters(), lr=learning_rate, betas=(0.9,0.99), eps=1e-10)

	def init_normal(m):
		if type(m) == nn.Linear:
			nn.init.kaiming_normal_(m.weight)

	net0.apply(init_normal)
	net1.apply(init_normal)
	net0.train()
	net1.train()

	############################################################################

	def criterion(x,mu,w,mu0,mu1,vt0,vt1,vt0b,vt1b,k,kb,L,Lb,vk,tau,C):
		x.requires_grad = True
		mu.requires_grad = True
		vt0.requires_grad = True
		vt1.requires_grad = True
		L.requires_grad = True
		k.requires_grad = True

		######### Interior points ##########
		e0_in = torch.cat((x,mu,k,vt0,L,torch.zeros_like(x)),1)
		e1_in = torch.cat((x,mu,k,vt1,L,torch.ones_like(x)),1)
		e0 = net0(e0_in)*(10**(vt0-L))*dT # Nonequilibrium part
		e1 = net0(e1_in)*(10**(vt1-L))*dT # Equilibrium part
		eEq = net1(torch.cat((x,L),1))*dT

		e0_x = torch.autograd.grad(e0+eEq,x,grad_outputs=torch.ones_like(x).to(device),create_graph=True)[0]
		e1_x = torch.autograd.grad(e1+eEq,x,grad_outputs=torch.ones_like(x).to(device),create_graph=True)[0]

		e = torch.cat(((e0+eEq).reshape(-1,Ns*Nk),(e0+eEq).reshape(-1,Ns*Nk),(e1+eEq).reshape(-1,Ns*Nk)),1).reshape(-1,1)
		sum_e = torch.matmul(e.reshape(-1,Ns),w).reshape(-1,1)
		deltaT = torch.matmul(sum_e.reshape(-1,Nk*Np),C*wk/tau*vk/(4*np.pi)).reshape(-1,1).repeat(1,Nk*Ns).reshape(-1,1)/torch.sum(C/tau*wk*vk)
		e_x = torch.cat((e0_x.reshape(-1,Ns*Nk),e0_x.reshape(-1,Ns*Nk),e1_x.reshape(-1,Ns*Nk)),1).reshape(-1,1)
		sum_ex = torch.matmul(e_x.reshape(-1,Ns),w*mu[0:Ns].reshape(-1,1)).reshape(-1,1)
		dq = torch.matmul(sum_ex.reshape(-1,Nk*Np),C*wk*vk**2/(4*np.pi)).reshape(-1,1)

		######### Isothermal boundary ##########
		c0_in = torch.cat((torch.ones_like(mu1),mu1,kb,vt0b,Lb,torch.zeros_like(mu1)),1)
		c1_in = torch.cat((torch.ones_like(mu1),mu1,kb,vt1b,Lb,torch.ones_like(mu1)),1)
		c0 = net0(c0_in)*(10**(vt0b-Lb))
		c1 = net0(c1_in)*(10**(vt1b-Lb))
		cEq = net1(torch.cat((torch.ones_like(Lb),Lb),1))
		ec = torch.cat((c0+cEq,c1+cEq),0)

		h0_in = torch.cat((torch.zeros_like(mu0),mu0,kb,vt0b,Lb,torch.zeros_like(mu0)),1)
		h1_in = torch.cat((torch.zeros_like(mu0),mu0,kb,vt1b,Lb,torch.ones_like(mu0)),1)
		h0 = net0(h0_in)*(10**(vt0b-Lb))
		h1 = net0(h1_in)*(10**(vt1b-Lb))
		hEq = net1(torch.cat((torch.zeros_like(Lb),Lb),1))
		eh = torch.cat((h0+hEq,h1+hEq),0)

		######### Loss ##########
		loss_1 = (mu*e0_x + e0/(10**vt0)*(10**L))/dT  # bte residual for branch 0
		loss_2 = (mu*e1_x + e1/(10**vt1)*(10**L))/dT  # bte residual for branch 1
		loss_3 = (deltaT - eEq)/dT
		loss_4 = (dq/TC)
		loss_5 = (ec + 1)
		loss_6 = (eh - 1)

		##############
		# MSE LOSS
		loss_f = nn.MSELoss()

		loss1 = loss_f(loss_1,torch.zeros_like(loss_1))
		loss2 = loss_f(loss_2,torch.zeros_like(loss_2))
		loss3 = loss_f(loss_3,torch.zeros_like(loss_3))
		loss4 = loss_f(loss_4,torch.zeros_like(loss_4))
		loss5 = loss_f(loss_5,torch.zeros_like(loss_5))
		loss6 = loss_f(loss_6,torch.zeros_like(loss_6))

		return loss1, loss2, loss3, loss4, loss5, loss6

	###################################################################

	# Main loop
	Loss_mean = 100
	Loss_list = []
	Loss_res_list = []
	tic = time.time()

	p = np.concatenate((np.zeros_like(k),np.zeros_like(k),np.ones_like(k)),0)
	v,tau,C = param(np.tile(k,(Np,1)),p,Tr)
	v = torch.FloatTensor(v).to(device)
	tau = torch.FloatTensor(tau).to(device)
	C = torch.FloatTensor(C).to(device)

	Nl = len(L)
	Lx = torch.FloatTensor(L).repeat(Nx,1).to(device)
	x = torch.FloatTensor(x).repeat(1,Ns*Nk*Nl).reshape(-1,1).to(device)
	mu0 = torch.FloatTensor(mu[mu>0].reshape(-1,1)).repeat(Nk*Nl,1).to(device)
	mu1 = torch.FloatTensor(mu[mu<0].reshape(-1,1)).repeat(Nk*Nl,1).to(device)
	mu = torch.FloatTensor(mu).repeat(Nx*Nk*Nl,1).to(device)
	w = torch.FloatTensor(w).to(device)
	wk = np.pi*2/a/Nk

	vt0b = torch.FloatTensor(vt0).repeat(1,int(Ns/2)).reshape(-1,1).repeat(Nl,1).to(device)
	vt1b = torch.FloatTensor(vt1).repeat(1,int(Ns/2)).reshape(-1,1).repeat(Nl,1).to(device)
	vt0 = torch.FloatTensor(vt0).repeat(1,Ns).reshape(-1,1).repeat(Nx*Nl,1).to(device)
	vt1 = torch.FloatTensor(vt1).repeat(1,Ns).reshape(-1,1).repeat(Nx*Nl,1).to(device)
	Lb = torch.FloatTensor(L).repeat(1,int(Ns/2)*Nk).reshape(-1,1).to(device)
	L = torch.FloatTensor(L).repeat(1,Ns*Nk).reshape(-1,1).repeat(Nx,1).to(device)

	k = k/(np.pi*2/a)
	kb = torch.FloatTensor(k).repeat(1,int(Ns/2)).reshape(-1,1).repeat(Nl,1).to(device)
	k = torch.FloatTensor(k).repeat(1,Ns).reshape(-1,1).repeat(Nx*Nl,1).to(device)
	TC = ((1/3)*torch.sum(C*v**3*tau*wk)/(dT*2)*1e11/(10**Lx)).to(device)

	for epoch in range(epochs):
		Loss = []
		Loss1 = []
		net0.zero_grad()
		net1.zero_grad()
		loss1,loss2,loss3,loss4,loss5,loss6 = criterion(x,mu,w,mu0,mu1,vt0,vt1,vt0b,vt1b,k,kb,L,Lb,v,tau,C)
		loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6
		loss.backward()
		optimizer0.step()
		optimizer1.step()
		Loss.append((loss1+loss2+loss3+loss4+loss5+loss6).item())
		Loss1.append((loss1+loss2).item())
		if epoch%2000 == 0:
			print('Train Epoch: {}  Loss: {:.6f}  {:.6f}  {:.6f}  {:.6f}'.format(epoch,loss1.item(),loss3.item(),loss5.item(),loss6.item()))
			# torch.save(net0.state_dict(),path+"train_ng_epoch"+str(epoch)+"e.pt")
		Loss = np.array(Loss)
		Loss1 = np.array(Loss1)
		if np.mean(Loss) < Loss_mean:
			checkpoint0 = {
				'epoch': epoch + 1,
				'state_dict': net0.state_dict(),
				'optimizer': optimizer0.state_dict(),
				'loss': loss.item()
			}
			checkpoint1 = {
				'epoch': epoch + 1,
				'state_dict': net1.state_dict(),
				'optimizer': optimizer1.state_dict(),
				'loss': loss.item()
			}
			torch.save(checkpoint0,path+"model0.pt")
			torch.save(checkpoint1,path+"model1.pt")
			Loss_mean = np.mean(Loss)
		Loss_list.append(np.mean(Loss))
		Loss_res_list.append(np.mean(Loss1))

	toc = time.time()
	elapseTime = toc - tic
	print ("elapse time in parallel = ", elapseTime)
	############################################################
	np.savetxt(path+'Loss.txt',np.array(Loss_list))
	np.savetxt(path+'Loss_res.txt',np.array(Loss_res_list))

