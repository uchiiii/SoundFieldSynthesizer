import numpy as np 
import scipy as sp
from scipy.special import *  
from ai import cs

class ModelMatchM:
    '''
    This is implemented with the iterior uniformly wighted L2 norm.
    '''  
    def __init__(self, r, r_c, r_s, Rint, gamma ,N):
        '''
        input
            r:(L,3) array which includes the position of all the L speakers on the cartician coordinate.
            r_c:(Q,3) array which includes the position of the centers of each area.
            r_s:()
            Rint:(Q) array of the redius of sphere.
            gamma:(Q) array of the weights of the each sphere.  
            N: the trunction orders.
        '''
        self.c = 343.0
        #self.k = omega/c
        self.N = N 
        self.Rint = Rint
        self.gamma = gamma
        self.L = r.shape[0]
        self.r = r
        self.r_c = r_c
        self.r_s = r_s
        self.nu = self.__get_index_harmonic()
        self.mu = np.array([j for i in range(0,self.N+1) for j in range(-i,i+1)] )
        print('nu={}'.format(self.nu))
        print('mu={}'.format(self.mu))

    def __hankel(self,n,z):
        '''spherical hankel function of the first kind'''
        return spherical_jn(n,z) - 1.0j*spherical_yn(n,z)

    def __get_index_harmonic(self):
        a = np.arange(0,self.N+1)
        return np.repeat(a, 2*a+1)
        
    def __c_element(self,k,nu,mu,r,theta,phi):
        return 1.0j*k*self.__hankel(nu, k*r) * np.conj(self.__sph_harm(mu, nu, theta, phi))
    
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
        nu_s = self.nu.T.reshape(row_num,1)
        mu_s = self.mu.T.reshape(row_num,1)
        r_s = np.tile(r, (row_num,1))
        theta_s = np.tile(theta, (row_num,1))
        phi_s = np.tile(phi, (row_num,1))
        return self.__c_element(k,nu_s,mu_s,r_s,theta_s,phi_s)

    def __spherical_jn_from_vessel(self,nu,x):
        return np.sqrt(np.pi/(2*x))*jv(nu+(1/2),x)

    def __sph_harm(self,mu,nu,theta,phi):
        return sph_harm(mu, nu, phi, theta)

    def __get_interior_weight(self,k,idx):
        kRint_s = k * self.Rint[idx] * np.ones_like(self.nu)
        W_uni = 2.0 * np.pi * self.Rint[idx]**3 * (spherical_jn(self.nu, kRint_s)**2 - self.__spherical_jn_from_vessel(self.nu-1, kRint_s) * spherical_jn(self.nu+1, kRint_s))
        #W_uni[0] = 2.0 * np.pi * self.Rint[idx]**3 * (spherical_jn(self.nu[0], kRint_s[0])**2 - self.__spherical_jn_from_vessel(self.nu[0]-1, kRint_s[0]) * spherical_jn(self.nu[0]+1, kRint_s[0]))
        return W_uni

    def __get_W(self,k,idx):
        W_uni = self.__get_interior_weight(k,idx) 
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
            r,theta,phi = self.__cart2sp(x=self.r[:,0]-self.r_c[idx,0], y=self.r[:,1]-self.r_c[idx,1], z=self.r[:,2]-self.r_c[idx,2])
            _r_c, _r_c_theta, _r_c_phi = self.__cart2sp(x=self.r_s[0]-self.r_c[idx,0], y=self.r_s[1]-self.r_c[idx,1], z=self.r_s[2]-self.r_c[idx,2])

            print(_r_c, _r_c_theta, _r_c_phi)
            W = self.__get_W(k, idx)
            print('W={}'.format(W))
            C = self.__get_C(k,r,theta,phi)
            g = self.__get_g(k,_r_c,_r_c_theta,_r_c_phi)
            A += gamma[idx]*np.dot(np.conj(C).T, np.dot(W, C))
            b += gamma[idx]*np.dot(np.conj(C).T, np.dot(W, g))
        return A,b

    def exploit_d(self,k):
        '''
        return d vector:(L,1) 2-dimensional array
        '''
        A,b = self.__get_A_and_b(k)
        print('A={}'.format(A))
        _,s,_ = np.linalg.svd(A)
        _lambda = s[0] * 1e-3
        return np.dot(np.linalg.inv(A + _lambda*np.eye(A.shape[0], dtype=np.complex)), b)

'''simulation'''

import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def hankel(n,z):
    '''spherical hankel function of the first kind'''
    return spherical_jn(n,z) - 1.0j*spherical_yn(n,z)

def G(x,y,z,k): 
    '''
    x:(M,M,M) and so on 
    '''
    abs_rel = np.sqrt(np.square(x)+np.square(y)+np.square(z))
    return 1.0j*k/(4*np.pi)*hankel(0, k*abs_rel)


def multipoint_sw(x,y,z,r_s,k):
    '''return p_syn(r,w=kc)'''
    x,y,z = np.meshgrid(x,y,z)
    val = G(x-r_s[0],y-r_s[1],z-r_s[2],k)
    return val

def multipoint_sw_weight(x,y,z,r_s,k,d):
    '''return p_syn(r,w=kc)'''
    if d.shape[0] != r_s.shape[0]:
        print("The size is different. Something wrong.")
        sys.exit()
    x,y,z = np.meshgrid(x,y,z)
    val = np.zeros_like(x,dtype=np.complex)
    for i in range(r_s.shape[0]):
        val += d[i,0]*G(x-r_s[i,0],y-r_s[i,1],z-r_s[i,2],k)
    return val



if __name__=='__main__':
    r = np.array([[1,1,1],[1,-1,1],[-1,1,1],[-1,-1,1],[1,1,-1],[1,-1,-1],[-1,1,-1],[-1,-1,-1]])
    N = 5
    Rint = np.array([0.7])
    r_c = np.array([[0,0,0]])
    r_s = np.array([-2,-2,0])
    gamma = np.array([1.0])
    omega = 2*np.pi*150
    c = 343.0
    test_mmm = ModelMatchM(r=r,r_c=r_c,r_s=r_s,Rint=Rint,gamma=gamma,N=N)
    d = test_mmm.exploit_d(k=omega/c)
    print(d)

    x1 = np.arange(-5,5,0.01)
    y1 = np.arange(-5,5,0.01)
    z1 = np.arange(0,1,1)

    fig, (axsyn, axdes) = plt.subplots(ncols=2, figsize=(10,4))

    z_draw = 0 #This is an index.
    '''desired part'''
    val_des = multipoint_sw(x1,y1,z1,r_s,k=omega/c)
    cont_des = axdes.pcolormesh(x1, y1, np.real(val_des[:,:,z_draw]))
    axdes.plot(r_s[0], r_s[1], 'or', label='desired microphone')
    axdes.Circle(r_c[:,0], r_c[:,1], Rint, color='b', fillstyle='none')
    cont_des.set_clim(-0.02,0.02)
    axdes.set_title('desired')
    axdes.set_xlabel('x')
    axdes.set_ylabel('y')

    '''synthesized part'''
    val_syn = multipoint_sw_weight(x1,y1,z1,r,k=omega/c,d=d)
    cont_syn = axsyn.pcolormesh(x1, y1, np.real(val_syn[:,:,z_draw]))
    axsyn.plot(r[:,0], r[:,1], 'or', label='position of loudspeakers')
    cont_syn.set_clim(-0.02,0.02)
    axsyn.set_title('synthesized')
    axsyn.set_xlabel('x')
    axsyn.set_ylabel('y')

    fig.colorbar(cont_syn)
    
    axdes.legend()
    axsyn.legend()
    plt.show()