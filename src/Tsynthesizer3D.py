import numpy as np 
import scipy as sp
from scipy.special import *  
import scipy.fftpack as spfft
from ai import cs
from tqdm import tqdm
import sys
import time
import cis
import librosa
from wavefile import write

import matplotlib.pyplot as plt
import matplotlib.patches as patches

class ModelMatchM:
    '''
    This is implemented with the iterior uniformly wighted L2 norm.
    '''  
    def __init__(self, r, r_c, r_s, Rint, gamma, is_silent):
        '''
        input
            r:(L,3) array which includes the position of all the L speakers on the cartician coordinate.
            r_c:(Q,3) array which includes the position of the centers of each area.
            r_s:()
            Rint:(Q) array of the redius of sphere.
            gamma:(Q) array of the weights of the each sphere.  
            is_silent:(Q) array,1 if an area is silent, 0 if it is not.
        '''
        self.c = 343.0
        self.Rint = Rint
        self.gamma = gamma
        self.is_silent=is_silent
        self.L = r.shape[0]
        self.r = r
        self.r_c = r_c
        self.r_s = r_s

    def create_wav(self,input_file,output_file,f_max,f_low,M):
        '''
        assume that input wave is mono
        '''
        print('START!')
        rate, data = cis.wavread(input_file)
        print('- data shape of input wav file : {}'.format(data.shape))
        omega_mx = 2*np.pi*f_max
        omega_low = 2*np.pi*f_low
        data = librosa.resample(data.T,rate,2*f_max)
        rate = omega_mx/np.pi
        
        filt = np.real(self.exploit_transfer_func_T(omega_mx=omega_mx,M=M,omega_low=omega_low))
        #filt = np.zeros((self.L,1000))

        if data.ndim == 2:
            length = data.shape[1]
            data = data[0,:]
        elif data.ndim == 1:
            length = data.shape[0]
        else:
            print('invalid wave file!!!')
            sys.exit(0)

        _ans = np.zeros((self.L,length))
        
        start = time.time()

        for i in range(self.L):
            _ans[i,:] =  np.convolve(data,filt[i,:],mode='full')[0:length]
        elapsed_time = time.time()-start
        print("- elapsed_time of covolution:{0}".format(elapsed_time) + "[sec]")

        start = time.time()
        test = librosa.resample(_ans[0,:],rate,44100)
        resampled = np.zeros((self.L,test.shape[0]))
        for i in range(self.L):
            resampled[i,:] = librosa.resample(_ans[i,:],rate,44100)
        cis.wavwrite(output_file,44100,resampled.T,self.L)
        #cis.wavwrite(output_file,44100,resampled[3:5,:].T,2)
        elapsed_time = time.time()-start
        print("- elapsed_time of writing a wav file({0}) :{1}".format(output_file,elapsed_time) + "[sec]")
        print('DONE!')
        
        
    def exploit_transfer_func_T(self,omega_mx=4000,M=512,omega_low=200):
        '''
        M: the even number of omega.
        return filter.
        '''
        start = time.time()
        #t = 0.5
        omega_s = np.linspace(0,omega_mx,M+1)
        d_s = np.zeros((self.L,2*M), dtype=np.complex)
        for i, omega in tqdm(enumerate(omega_s)):
            if omega == 0:
                continue
            elif omega < omega_low:
                continue
            k = omega/self.c
            d = self.__exploit_d(k=k).flatten()
            #d = np.exp(-1j*omega*t)*np.ones(self.L)
            d_s[:,M-1+i] = d
            if i != M:
                d_s[:,M-1-i] = np.conj(d)
        d_s = np.append(d_s,d_s[:,0:M-1],axis=1)[:,M-1:]
        ans = spfft.ifft(d_s,axis=1)
        ans = np.append(ans,ans[:,0:M],axis=1)[:,M:]

        elapsed_time = time.time()-start
        print("- elapsed_time of filter:{0}".format(elapsed_time) + "[sec]")
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
            W_uni = self.__get_interior_weight(k,idx,nu)
            C = self.__get_C(k,r,theta,phi,nu,mu)
            g = self.__get_g(k,_r_c,_r_c_theta,_r_c_phi,nu,mu)
            A += self.gamma[idx]*np.dot(np.conj(C).T*W_uni, C)
            if self.is_silent[idx] == 0:
                b += self.gamma[idx]*np.dot(np.conj(C).T*W_uni, g)
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
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    '''
    NUM_L = 12 #the number of the used loudspeakers
    r = np.zeros((NUM_L,3))
    r[:,0] = -2
    if int((NUM_L/2)) != NUM_L/2:
        print('The number of the used loudspeakers is supposed to be even on this code.')
        sys.exit()
    r[:,2] = np.array([-0.2,0.2]*int((NUM_L/2))) 
    r[:,1] = np.linspace(-2.4,2.4,NUM_L)
    '''
    NUM_L = 9 #the number of the used loudspeakers
    r = np.zeros((NUM_L,3))
    x_diff = 44.5/2.0*1.0e-2
    y_diff = 44.5*np.sqrt(3)/6*1.0e-2
    z_diff = (119-102.5)*1.0e-2
    r[:,0] = np.arange(0,x_diff*NUM_L,x_diff)
    for i in range(NUM_L):
        if i%2 == 0:
            r[i,1] = 0.0
            r[i,2] = -z_diff/2
        else:
            r[i,1] = y_diff
            r[i,2] = z_diff/2
    Rint = np.array([0.5])
    is_silent = np.array([0])
    r_c = np.array([[1.0,1.3,0]]) #the center of target sphere 
    r_s = np.array([-2,1.3,0]) #the desired position of speaker
    gamma = np.array([1.0])
    omega = 2*np.pi*150
    c = 343.0
    test_mmm = ModelMatchM(r=r,r_c=r_c,r_s=r_s,Rint=Rint,gamma=gamma,is_silent=is_silent)
    M = 512
    omega_mx = 8000
    val = test_mmm.exploit_transfer_func_T(omega_mx=omega_mx,M=M)
    print(val.shape)
    for i in range(NUM_L):
        plt.plot(np.arange(0,(2*np.pi)/(omega_mx/M),(np.pi)/omega_mx) ,np.real(val)[i,:]) 
        #plt.plot(np.real(val[i,:]))
    plt.title('time domain transfer function ')
    plt.xlabel('t [s]')
    #plt.set_ylabel('')
    #plt.legend()
    plt.show()