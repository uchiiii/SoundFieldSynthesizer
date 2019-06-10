import numpy as np 
import scipy as sp
from scipy.special import *  
from ai import cs

class ModelMatchM:
    '''
    This is implemented with the iterior uniformly wighted L2 norm.
    '''  
    def __init__(r=r, N=N, Rint=Rint, omega=omega):
        '''
        input:
        r:(3,L) array which includes the position of all the L speakers on the cartician coordinate.
        N: the trunction orders.
        '''
        self.c = 343.0
        #self.k = omega/c
        self.N = N 
        self.Rint = Rint
        self.L = r.shape[1]
        self.r, self.theta, self.phi = cs.cart2sp(x=r[0,:], y=r[1,:], z=r[2,:])
        self.nu = self.__get_index_harmonic()
        self.mu = np.array([j for i in range(1,self.N+1) for j in range(-i,i+1)] )
        

    def __hankel(n,z):
        '''spherical hankel function of the first kind'''
        return spherical_jn(n,z) - spherical_yn(n,z)

    def __get_index_harmonic():
        a = np.arange(1,self.N+1)
        return np.repeat(a, 2*a+1)
        
    def __c_element(k,nu,mu,r,theta,phi):
        return 1.0j*k*self.__hankel(nu, k*r) * np.conj(sph_harm(mu, nu, theta, phi))
    
    def __get_C(k):
        row_num = mu.shape[0]
        nu_s = np.tile(self.nu, (L,1)).T
        mu_s = np.tile(self.mu, (L,1)).T
        r_s = np.tile(self.r, (row_num,1))
        theta_s = np.tile(self.theta, (row_num,1))
        phi_s = np.tile(self.phi, (row_num,1))
        return self.__c_element(k,nu_s,mu_s,r_s,theta_s,phi_s)

    def __get_g(k,idx):
        row_num = mu.shape[0]
        nu_s = self.nu.T
        mu_s = self.mu.T
        r_s = np.tile(self.r[idx], (row_num,1))
        theta_s = np.tile(self.theta[idx], (row_num,1))
        phi_s = np.tile(self.phi[idx], (row_num,1))
        return self.__c_element(k,nu_s,mu_s,r_s,theta_s,phi_s)

    def __get_W(k, omega):
        kRint_s = k * self.Rint * np.ones_like(self.nu)
        W_uni = 2.0 * np.pi * self.Rint**3 * (spherical_jn(self.nu, kRint_s)**2 - spherical_jn(self.nu-1, kRint_s) * spherical_jn(self.nu+1, kRint_s))
        return np.diag(W_uni)

    def __get_A_and_b(k,idx,omega):
        '''
        input(omega): specific omega (This should be revised afterwards.)
        return A, b
        '''
        W = self.__get_W(k,omega)
        C = self.__get_C(k)
        g = self.__get__g(k,idx)
        return np.dot(np.conj(C), np.dot(W, C)), np.dot(np.conj(C), np.dot(W, g))

    def __get_d_vec(k,omega):
        A,b = self.__get_A_and_b(k,idx, omega)
        U,s,V = np.linalg.svd(A)
        _lambda = s[0] * 1e-3
        return np.dot(np.linalg.inv(A + _lambda*np.eye(A.shape[0], dtype=np.complex)))
        