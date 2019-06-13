import numpy as np
import sys 
from scipy.special import *
from synthesizer3D import *
import matplotlib.pyplot as plt

def hankel(n,z):
    '''spherical hankel function of the first kind'''
    return spherical_jn(n,z) - spherical_yn(n,z)

def G(x,y,z,k): 
    '''
    x:(M,M,M) and so on 
    '''
    abs_rel = np.sqrt(np.square(x)+np.square(y)+np.square(z))
    return 1.0j*k/(4*np.pi)*hankel(0, k*abs_rel)


def mutipoint_sw(r,r_s,k):
    ''' return wave at r which are generated from point sources at r_c.
    r:(M^3,3) where M is the number of grids.
    r_s:(L,3) the coordi'nate of the speakers.
    '''
    rel = np.repeat(r,r_s.shape[0]).reshape(r.shape[0],r.shape[1],r_s.shape[0])
    abs_rel = np.sqrt(np.sum(np.square(rel),axis=1))
    return 1.0j*k/(4*np.pi)*hankel(0, k*abs_rel)

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
    N = 10
    Rint = np.array([0.7])    #r_c = np.array([[0+1e-4,0+1e-4,0+1e-4]])
    r_c = np.array([[0,0,0]])
    r_s = np.array([2,2,2])
    gamma = np.array([1.0])
    omega = 2*np.pi*150
    c = 343.0
    test_mmm = ModelMatchM(r=r,r_c=r_c,r_s=r_s,Rint=Rint,gamma=gamma,N=N)
    d = test_mmm.exploit_d(k=omega/c)
    print(d)

    x1 = np.arange(-2,2,0.01)
    y1 = np.arange(-2,2,0.01)
    z1 = np.arange(-2,2,0.01)

    val = multipoint_sw_weight(x1,y1,z1,r,k=omega/c,d=d)

    plt.contourf(x1, y1, val[:,:,200])
    plt.show()


    
