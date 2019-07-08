from Tsynthesizer3D import ModelMatchM

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
    LENGTH_OF_TRIANGLE = 44.5 #44.5
    x_diff = LENGTH_OF_TRIANGLE/2.0*1.0e-2
    y_diff = LENGTH_OF_TRIANGLE*np.sqrt(3)/6*1.0e-2
    z_diff = (119-102.5)*1.0e-2
    #r[:,0] = np.arange(0,x_diff*NUM_L,x_diff)
    r[:,0] = np.linspace(0,x_diff*(NUM_L-1),NUM_L)
    for i in range(NUM_L):
        if i%2 == 0:
            r[i,1] = 0.0
            r[i,2] = -z_diff/2
        else:
            r[i,1] = -y_diff
            r[i,2] = z_diff/2

    Rint = np.array([0.2,0.05])
    is_silent = np.array([1, 0])
    r_c = np.array([[0.5 ,1.4 ,0],[1.5,1.4,0]]) #the center of target sphere 
    r_s = np.array([1.0,-2.0,0]) #the desired position of speaker
    gamma = np.array([5.0,1.0])
    '''
    Rint = np.array([0.5])
    is_silent = np.array([0])
    r_c = np.array([[0.5,1.4, 0]]) #the center of target sphere 
    r_s = np.array([0,0,-z_diff]) #the desired position of speaker
    gamma = np.array([1.0])
    '''
    print('r = {}'.format(r))
    c = 343.0
    test_mmm = ModelMatchM(r=r,r_c=r_c,r_s=r_s,Rint=Rint,gamma=gamma,is_silent=is_silent)
    M = 512
    f_max = 1000
    f_low = 500

    #input_file='./tests/asano.wav'
    #input_file='./tests/maracas.wav'
    input_file='./tests/whitenoise.wav'
    output_file='./tests/created_multi_white.wav' 
    #output_file='./tests/created_single.wav' 
    test_mmm.create_wav(input_file,output_file,f_max=f_max,f_low=f_low,M=M)



