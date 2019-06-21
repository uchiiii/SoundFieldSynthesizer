import numpy as np
import sys 
from scipy.special import *
from synthesizer3D import ModelMatchM
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

def squared_error_ratio(val_des, val_syn):
    diff = np.abs(val_syn - val_des)**2
    return 10*np.log(diff/(np.abs(val_des)**2))


if __name__=='__main__':
    NUM_L = 12 #the number of the used loudspeakers
    r = np.zeros((NUM_L,3))
    r[:,0] = -2
    if int((NUM_L/2)) != NUM_L/2:
        print('The number of the used loudspeakers is supposed to be even on this code.')
        sys.exit()
    r[:,2] = np.array([-0.2,0.2]*int((NUM_L/2))) 
    r[:,1] = np.linspace(-2.4,2.4,NUM_L)
    N = 5
    Rint = np.array([0.5,0.5]) 
    r_c = np.array([[0,0,0],[4,-4,0]]) #the center of target sphere 
    r_s = np.array([-3,0,0]) #the desired position of speaker
    gamma = np.array([1.0,1.0])
    omega = 2*np.pi*150
    c = 343.0
    test_mmm = ModelMatchM(r=r,r_c=r_c,r_s=r_s,Rint=Rint,gamma=gamma,N=N)
    d = test_mmm.exploit_d(k=omega/c)
    print(d)

    x1 = np.arange(-5,5,0.1)
    y1 = np.arange(-5,5,0.1)
    z1 = np.arange(0,1,1)

    fig, (axsyn, axdes) = plt.subplots(ncols=2, figsize=(9,4), sharey=True)

    z_draw = 0 #This is an index.
    '''desired part'''
    val_des = multipoint_sw(x1,y1,z1,r_s,k=omega/c)
    cont_des = axdes.pcolormesh(x1, y1, np.real(val_des[:,:,z_draw]))
    axdes.plot(r_s[0], r_s[1], 'or', label='desired microphone')
    for i in range(gamma.shape[0]):
        disk1 = plt.Circle((r_c[i,0],r_c[i,1]), Rint[i], color='k', fill=False, linestyle='dashed')
        axdes.add_artist(disk1)
    cont_des.set_clim(-0.02,0.02)
    axdes.set_title('desired')
    axdes.set_aspect('equal', 'box')
    axdes.set_xlabel('x[m]')
    axdes.set_ylabel('y[m]')

    '''synthesized part'''
    val_syn = multipoint_sw_weight(x1,y1,z1,r,k=omega/c,d=d)
    cont_syn = axsyn.pcolormesh(x1, y1, np.real(val_syn[:,:,z_draw]))
    axsyn.plot(r[:,0], r[:,1], 'or', label='position of loudspeakers')
    for i in range(gamma.shape[0]):
        disk2 = plt.Circle((r_c[i,0],r_c[i,1]), Rint[i], color='k', fill=False,linestyle='dashed')
        axsyn.add_artist(disk2)
    cont_syn.set_clim(-0.02,0.02)
    axsyn.set_title('synthesized')
    axsyn.set_aspect('equal', 'box')
    axsyn.set_xlabel('x[m]')
    axsyn.set_ylabel('y[m]')

    fig.colorbar(cont_syn)
    fig.tight_layout()
    axdes.legend()
    axsyn.legend()

    fig1, axerror = plt.subplots(ncols=1, figsize=(5,4), sharey=True)    
    '''error part'''
    val_error = squared_error_ratio(val_des, val_syn)
    cont_error = axerror.pcolormesh(x1, y1, np.real(val_error[:,:,z_draw]))
    axerror.plot(r[:,0], r[:,1], 'or', label='position of loudspeakers')
    for i in range(gamma.shape[0]):
        disk3 = plt.Circle((r_c[i,0],r_c[i,1]), Rint[i], color='k', fill=False,linestyle='dashed')
        axerror.add_artist(disk3)
    cont_error.set_clim(-50,0)
    axerror.set_title('NMSE')
    axerror.set_aspect('equal', 'box')
    axerror.set_xlabel('x[m]')
    axerror.set_ylabel('y[m]')

    fig1.colorbar(cont_error)
    axerror.legend()

    plt.show()
    
