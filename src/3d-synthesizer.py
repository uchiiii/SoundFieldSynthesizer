import numpy as np 
import scipy as sp
from scipy.special import *  
from ai import cs

class ModelMatchM:
    '''
    This is implemented with the iterior uniformly wighted L2 norm.
    '''  
    def __init__(r=r, N=N, omega=omega):
        '''
        input:
        r:(3,L) array which includes the position of all the L speakers on the cartician coordinate.
        N: the trunction orders.
        '''
        self.c = 343.0
        self.k = omega/c
        self.N = N 
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
        
    def __c_element(nu,mu,r,theta,phi):
        return self.__hankel(nu, self.k*r) * sph_harm(mu, nu, theta, phi)
    
    def __get_C():
        row_num = mu.shape[0]
        nu_s = np.tile(self.nu, (L,1)).T
        mu_s = np.tile(self.mu, (L,1)).T
        r_s = np.tile(self.r, (row_num,1))
        theta_s = np.tile(self.theta, (row_num,1))
        phi_s = np.tile(self.phi, (row_num,1))
        return self.__c_element(nu_s,mu_s,r_s,theta_s,phi_s)

    def __get_g(idx):
        row_num = mu.shape[0]
        nu_s = self.nu.T
        mu_s = self.mu.T
        r_s = np.tile(self.r[idx], (row_num,1))
        theta_s = np.tile(self.theta[idx], (row_num,1))
        phi_s = np.tile(self.phi[idx], (row_num,1))
        return self.__c_element(nu_s,mu_s,r_s,theta_s,phi_s)


    def __get_d_vec(A, b):
        U,s,V = np.linalg.svd(A)
        _lambda = s[0] * 1e-3
        return np.dot(np.linalg.inv(A + _lambda*np.eye(A.shape[0], dtype=np.complex)))

    def __get_A():
        

    def __get_b():
    
        