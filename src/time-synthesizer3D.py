import numpy as np 
import scipy as sp
from scipy.special import *  
import scipy.fftpack as spfft
from ai import cs
from tqdm import tqdm
import sys
import time

class ModelMatchM:
    '''
    This is implemented with the iterior uniformly wighted L2 norm.
    '''  
    def __init__(self, r, r_c, r_s, Rint, gamma):
        '''
        input
            r:(L,3) array which includes the position of all the L speakers on the cartician coordinate.
            r_c:(Q,3) array which includes the position of the centers of each area.
            r_s:()
            Rint:(Q) array of the redius of sphere.
            gamma:(Q) array of the weights of the each sphere.  
        '''
        self.c = 343.0
        self.Rint = Rint
        self.gamma = gamma
        self.L = r.shape[0]
        self.r = r
        self.r_c = r_c
        self.r_s = r_s
        
    def exploit_transfer_func_T(self,omega_mx=4000,M=512):
        '''
        M: the number of omega.
        '''
        start = time.time()
        omega_s = np.linspace(0,omega_mx,M+1)
        d_s = np.zeros((self.L,2*M), dtype=np.complex)
        for i, omega in tqdm(enumerate(omega_s)):
            if omega == 0:
                continue
            k = omega/self.c
            d = self.__exploit_d(k=k).flatten()
            d_s[:,M-1+i] = d
            if i != M:
                d_s[:,M-1-i] = np.conj(d)
        ans = np.zeros_like(d_s)
        for i in range(self.L):
            ans[i,:] = spfft.ifft(d_s[i,:])
        
        elapsed_time = time.time()-start
        print("elapsed_time:{0}".format(elapsed_time) + "[sec]")
        return ans
        
    def __hankel(self,n,z):
        '''spherical hankel function of the first kind'''
        return spherical_jn(n,z) - 1.0j*spherical_yn(n,z)

    def __get_index_harmonic(self,N):
        a = np.arange(0,N+1)
        return np.repeat(a, 2*a+1)
        
    def __c_element(self,k,nu,mu,r,theta,phi):
        return 1.0j*k*self.__hankel(nu, k*r) * np.conj(self.__sph_harm(mu, nu, theta, phi))
    
    def __get_C(self,k,r,theta,phi,nu,mu):
        row_num = mu.shape[0]
        nu_s = np.tile(nu, (self.L,1)).T
        mu_s = np.tile(mu, (self.L,1)).T
        r_s = np.tile(r, (row_num,1))
        theta_s = np.tile(theta, (row_num,1))
        phi_s = np.tile(phi, (row_num,1))
        return self.__c_element(k,nu_s,mu_s,r_s,theta_s,phi_s)

    def __get_g(self,k,r,theta,phi,nu,mu):
        row_num = mu.shape[0]
        nu_s = nu.T.reshape(row_num,1)
        mu_s = mu.T.reshape(row_num,1)
        r_s = np.tile(r, (row_num,1))
        theta_s = np.tile(theta, (row_num,1))
        phi_s = np.tile(phi, (row_num,1))
        return self.__c_element(k,nu_s,mu_s,r_s,theta_s,phi_s)

    def __spherical_jn_from_vessel(self,nu,x):
        return np.sqrt(np.pi/(2*x))*jv(nu+(1/2),x)

    def __sph_harm(self,mu,nu,theta,phi):
        return sph_harm(mu, nu, phi, theta)

    def __get_interior_weight(self,k,idx,nu):
        kRint_s = k * self.Rint[idx] * np.ones_like(nu)
        W_uni = 2.0 * np.pi * self.Rint[idx]**3 * (spherical_jn(nu, kRint_s)**2 - self.__spherical_jn_from_vessel(nu-1, kRint_s) * spherical_jn(nu+1, kRint_s))
        return W_uni

    def __get_W(self,k,idx,nu):
        W_uni = self.__get_interior_weight(k,idx,nu) 
        return np.diag(W_uni)

    def __cart2sp(self,x,y,z):
        r,theta,phi = cs.cart2sp(x,y,z)
        theta[np.isnan(theta)] = 0
        phi[np.isnan(phi)] = 0
        return r,np.pi/2-theta,phi

    def __get_A_and_b(self,k):
        '''
        input(idx): represents r_c[idx]
        return A, b
        '''
        A = np.zeros((self.L,self.L), dtype=np.complex)
        b = np.zeros((self.L,1), dtype=np.complex)
        for idx in range(self.gamma.shape[0]):
            N = int(np.minimum(int(3.0*k*self.Rint[idx]+1), 10)) #N: the trunction orders.
            nu = self.__get_index_harmonic(N)
            mu = np.array([j for i in range(0,N+1) for j in range(-i,i+1)] )
            r,theta,phi = self.__cart2sp(x=self.r[:,0]-self.r_c[idx,0], y=self.r[:,1]-self.r_c[idx,1], z=self.r[:,2]-self.r_c[idx,2])
            _r_c, _r_c_theta, _r_c_phi = self.__cart2sp(x=self.r_s[0]-self.r_c[idx,0], y=self.r_s[1]-self.r_c[idx,1], z=self.r_s[2]-self.r_c[idx,2])
            W = self.__get_W(k,idx,nu)
            C = self.__get_C(k,r,theta,phi,nu,mu)
            g = self.__get_g(k,_r_c,_r_c_theta,_r_c_phi,nu,mu)
            A += self.gamma[idx]*np.dot(np.conj(C).T, np.dot(W, C))
            b += self.gamma[idx]*np.dot(np.conj(C).T, np.dot(W, g))
        return A,b

    def __exploit_d(self,k):
        '''
        return d vector:(L,1) 2-dimensional array
        '''
        A,b = self.__get_A_and_b(k)
        _,s,_ = np.linalg.svd(A)
        _lambda = s[0] * 1e-3
        return np.dot(np.linalg.inv(A + _lambda*np.eye(A.shape[0], dtype=np.complex)), b)

if __name__=='__main__':
    #r = np.array([[-2,1,0.2],[-2,-1,0.2],[-2,1,0.2],[-2,-1,0.2],[-2,1,-0.2],[-2,-1,-0.2],[-2,1,-0.2],[-2,-1,-0.2]])
    NUM_L = 12 #the number of the used loudspeakers
    r = np.zeros((NUM_L,3))
    r[:,0] = -2
    if int((NUM_L/2)) != NUM_L/2:
        print('The number of the used loudspeakers is supposed to be even on this code.')
        sys.exit()
    r[:,2] = np.array([-0.2,0.2]*int((NUM_L/2))) 
    r[:,1] = np.linspace(-2.4,2.4,NUM_L)
    N = 5
    Rint = np.array([0.7]) 
    r_c = np.array([[0,0,0]]) #the center of target sphere 
    r_s = np.array([-3,0,0]) #the desired position of speaker
    gamma = np.array([1.0])
    omega = 2*np.pi*150
    c = 343.0
    test_mmm = ModelMatchM(r=r,r_c=r_c,r_s=r_s,Rint=Rint,gamma=gamma)
    val = test_mmm.exploit_transfer_func_T(M=512)
    print(val.shape)
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    for i in range(12):
        plt.plot(val[i,:])
    plt.show()