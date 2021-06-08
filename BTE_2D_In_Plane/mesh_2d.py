import numpy as np
import torch
hbar = 1
hkB = (1.054572/1.380649)*1e2
a = 5.431 #Angstrom

def TwoD_train_mesh(Nx,N1,N2,Nk):
	soboleng = torch.quasirandom.SobolEngine(dimension=2)
	mesh = soboleng.draw(Nx).numpy()
	# mesh = sobol_seq.i4_sobol_generate(2, Nx)
	x = mesh[:,0].reshape(-1,1)
	y = mesh[:,1].reshape(-1,1)

	mu, w1 = np.polynomial.legendre.leggauss(N1)
	phi, w2 = np.polynomial.legendre.leggauss(N2)
	phi = (phi + 1) * np.pi/2 

	mu = np.tile(mu.reshape(-1,1),(1,N2)).reshape(-1,1)
	w1 = np.tile(w1.reshape(-1,1),(1,N2)).reshape(-1,1)
	phi = np.tile(phi.reshape(-1,1),(N1,1))
	w2 = np.tile(w2.reshape(-1,1),(N1,1))
	
	eta = np.sqrt(1 - mu**2) * np.cos(phi)
	w = w1*w2*np.pi

	k = np.linspace(0,1,Nk*2+1)[1:Nk*2+1].reshape(-1,1)
	k = k[np.arange(0,Nk*2-1,2)]*np.pi*2/a

	return x,y,mu,eta,w,k

def Boundary_diffuse(mu,eta,w):
	s = np.concatenate((mu,eta),1)
	s1 = s[s[:,1]<0]
	w1 = w[eta<0].reshape(-1,1)
	s2 = s[s[:,1]>0]
	w2 = w[eta>0].reshape(-1,1)
	scale = -np.sum(eta[eta<0].reshape(-1,1)*w1)

	return s1,s2,w1,w2,scale

def TwoD_test_mesh(Nx,Ny,N1,N2,Nk):
	x = np.linspace(0,1,Nx+2)[1:Nx+1]
	y,wy = np.polynomial.legendre.leggauss(Ny)
	y = (y+1)/2
	x = np.tile(x.reshape(-1,1),(Ny,1))
	y = np.tile(y.reshape(-1,1),(1,Nx)).reshape(-1,1)
	wy = np.reshape(wy/2,(-1,1))

	mu, w1 = np.polynomial.legendre.leggauss(N1)
	phi, w2 = np.polynomial.legendre.leggauss(N2)
	phi = (phi + 1) * np.pi/2 

	mu = np.tile(mu.reshape(-1,1),(1,N2)).reshape(-1,1)
	w1 = np.tile(w1.reshape(-1,1),(1,N2)).reshape(-1,1)
	phi = np.tile(phi.reshape(-1,1),(N1,1))
	w2 = np.tile(w2.reshape(-1,1),(N1,1))
	
	eta = np.sqrt(1 - mu**2) * np.cos(phi)
	w = w1*w2*np.pi

	k = np.linspace(0,1,num=Nk*2+1)[1:Nk*2+1].reshape(-1,1)
	k = k[np.arange(0,Nk*2-1,2)]*np.pi*2/a

	return x,y,wy,mu,eta,w,k

# use reference temperature
def param(k,p,T):
	c10 = 5.23 #TA 1e13 A/s
	c20 = -2.26
	c11 = 9.01 #LA
	c21 = -2.0
	Ai = 1.498e7
	BL = 1.18e2
	BT = 8.708
	BU = 2.89e8

	c1 = (c11-c10)*p + c10
	c2 = (c21-c20)*p + c20
	om = c1*k + c2*k**2 # unit 1e13 1/s
	v = c1 + 2*c2*k # unit 1e13 A/s

	step = np.heaviside(k-np.pi/a,1)

	ti = Ai*om**4
	t1 = BL*om**2*T**3
	t01 = BT*om*T**4
	t02 = BU*om**2/np.sinh(hkB*om/T)
	t0 = t01*(1-step) + t02*step
	tNU = (t1-t0)*p + t0
	tau = 1/(ti+tNU) # s

	dfeq = hkB*om/(T**2)*np.exp(hkB*om/T)/(np.exp(hkB*om/T)-1)**2
	D = k**2/(2*np.pi**2*v) # 1e-13 s/A^3 
	C = hbar*om*D*dfeq # Js/A^3/K

	return v,tau,C

def TwoD_vt(k,Tr):
	v0,tau0,_ = param(k,np.zeros_like(k),Tr)
	v1,tau1,_ = param(k,np.ones_like(k),Tr)
	vt0 = np.log10(v0*tau0*1e11)
	vt1 = np.log10(v1*tau1*1e11)

	return vt0,vt1
