import numpy as np
import sys 
from scipy.special import *
import synthesizer3D

def hankel(n,z):
    '''spherical hankel function of the first kind'''
    return spherical_jn(n,z) - spherical_yn(n,z)

def mutipoint_sw(r,r_s,k):
    ''' return wave at r which are generated from point sources at r_c.
    r:(3)
    r_s:(L,3) the coordi'nate of the speakers.
    '''
    rel = r_s - r
    abs_rel = np.sqrt(np.sum(np.square(rel),axis=1))
    return 1.0j*k/(4*np.pi)*hankel(0, k*abs_rel)

    
def multipoint_sw_weight(r,r_s,k,d):
    if d.shape[0] != r_s.shape[0]:
        print("The size is different. Something wrong.")
        sys.exit()
    return d*mutipoint_sw(r,r_s,k)

