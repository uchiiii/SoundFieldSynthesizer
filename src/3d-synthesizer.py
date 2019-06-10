import numpy as np 
import scipy as sp
from scipy.special import *  
from ai import cs

class ModelMatchM:
    '''
    This is implemented with the iterior uniformly wighted L2 norm.
    '''  
    def __init__(self, r, r_c, Rint, gamma ,N):
        '''
        input
            r:(L,3) array which includes the position of all the L speakers on the cartician coordinate.
            r_c:(Q,3) array which includes the position of the centers of each area.
            Rint:(Q) array of the redius of sphere.
            gamma:(Q) array of the weights of the each sphere.  
            N: the trunction orders.
        '''
        self.c = 343.0
        #self.k = omega/c
        self.N = N 
        self.Rint = Rint
        self.gamma = gamma
        self.L = r.shape[1]
        self.r = r
        self.r_c = r_c
        #self.r, self.theta, self.phi = cs.cart2sp(x=r[:,0], y=r[:,1], z=r[:,2])
        self.nu = self.__get_index_harmonic()
        self.mu = np.array([j for i in range(1,self.N+1) for j in range(-i,i+1)] )

    def __hankel(self,n,z):
        '''spherical hankel function of the first kind'''
        return spherical_jn(n,z) - spherical_yn(n,z)

    def __get_index_harmonic(self):
        a = np.arange(1,self.N+1)
        return np.repeat(a, 2*a+1)
        
    def __c_element(self,k,nu,mu,r,theta,phi):
        return 1.0j*k*self.__hankel(nu, k*r) * np.conj(sph_harm(mu, nu, theta, phi))
    
    def __get_C(self,k,r,theta,phi):
        row_num = self.mu.shape[0]
        nu_s = np.tile(self.nu, (self.L,1)).T
        mu_s = np.tile(self.mu, (self.L,1)).T
        r_s = np.tile(r, (row_num,1))
        theta_s = np.tile(theta, (row_num,1))
        phi_s = np.tile(phi, (row_num,1))
        return self.__c_element(k,nu_s,mu_s,r_s,theta_s,phi_s)

    def __get_g(self,k,r,theta,phi):
        row_num = self.mu.shape[0]
        nu_s = self.nu.T
        mu_s = self.mu.T
        r_s = np.tile(r, (row_num,1))
        theta_s = np.tile(theta, (row_num,1))
        phi_s = np.tile(phi, (row_num,1))
        return self.__c_element(k,nu_s,mu_s,r_s,theta_s,phi_s)

    def __get_W(self,k,idx):
        kRint_s = k * self.Rint[idx] * np.ones_like(self.nu)
        W_uni = 2.0 * np.pi * self.Rint[idx]**3 * (spherical_jn(self.nu, kRint_s)**2 - spherical_jn(self.nu-1, kRint_s) * spherical_jn(self.nu+1, kRint_s))
        return np.diag(W_uni)

    def __get_A_and_b(self,k):
        '''
        input(idx): represents r_c[idx]
        return A, b
        '''
        A = np.zeros((self.nu.shape[0],self.nu.shape[0]))
        b = np.zeros_like(self.nu)
        for idx in range(self.gamma.shape[0]):
            r,theta,phi = cs.cart2sp(x=self.r[:,0]-self.r_c[idx,0], y=self.r[:,1]-self.r_c[idx,1], z=self.r[:,2]-self.r_c[idx,2])
            _r_c, _r_c_theta, _r_c_phi = cs.cart2sp(x=self.r_c[idx,0], y=self.r_c[idx,1], z=self.r_c[idx,2])
            W = self.__get_W(k, idx)
            C = self.__get_C(k,r,theta,phi)
            g = self.__get_g(k,_r_c,_r_c_theta,_r_c_phi)
            A += gamma[idx]*np.dot(np.conj(C), np.dot(W, C))
            b += gamma[idx]*np.dot(np.conj(C), np.dot(W, g))
        return A,b

    def __get_d_vec(self,k):
        A,b = self.__get_A_and_b(k)
        _,s,_ = np.linalg.svd(A)
        _lambda = s[0] * 1e-3
        return np.dot(np.linalg.inv(A + _lambda*np.eye(A.shape[0], dtype=np.complex)), b)

if __name__=='__main__':
    '''test'''
    r = np.array([[1,1,1],[1,-1,1],[-1,1,1],[-1,-1,-1]])
    N = 10
    Rint = np.array([0.4])
    r_c = np.array([[0,0,0]])
    gamma = np.array([1.0])
    omega = 150
    test_mmm = ModelMatchM(r=r,r_c=r_c,Rint=Rint,gamma=gamma,N=N)