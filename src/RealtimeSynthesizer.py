import sys
import time
import pyaudio
import wave
import time
import threading
import numpy as np
import scipy as sp
from scipy.special import *
from scipy import signal
import scipy.fftpack as spfft
from ai import cs
import librosa
from wavefile import write

class RealtimeSynthesizer:
    def __init__(self,fname,r, r_c, r_s, Rint, gamma, is_silent,f_max=4000,f_low=200,M=512,start_channel=1,Fs=0,dev_id=-1):
        '''
        input
            r:(L,3) array which includes the position of all the L speakers on the cartician coordinate.
            r_c:(Q,3) array which includes the position of the centers of each area.
            r_s:()
            Rint:(Q) array of the redius of sphere.
            gamma:(Q) array of the weights of the each sphere.  
            is_silent:(Q) array,1 if an area is silent, 0 if it is not.
            f_max: maximum of the frequency considered.
            f_min: minimum of the frequency considered.
            M: filter length.
        '''

        '''filter part'''
        self.c = 343.0
        self.Rint = Rint
        self.gamma = gamma
        self.is_silent=is_silent
        self.L = r.shape[0]
        self.r = r
        self.r_c = r_c
        self.r_s = r_s
        self.f_max = f_max
        self.f_low = f_low
        self.M = M
        self.omega_mx = 2*np.pi*self.f_max
        self.omega_low = 2*np.pi*self.f_low
        self.filt = np.zeros((self.L,2*self.M))

        self.lock = threading.Lock()

        '''pyaudio part'''
        self.chunk = 32768 #length of chunk for pyaudio
        self.format = pyaudio.paInt16 #format

        self.out_fname = fname # output filename
        self.start_channel = start_channel # number of input channels
        self.Fs = Fs #sampling frequency
        self.dev_id = dev_id #index of audio device

        self.wf_out = wave.open(self.out_fname, 'rb')

        # Sampling frequency
        if self.Fs<=0:
            self.Fs = int(self.wf_out.getframerate())

        # Number of channels of input wav
        self.n_out_channel = int(self.wf_out.getnchannels())
        if self.n_out_channel > 2:
            print('input wav file needs to be monoral or stereo.')
            sys.exit(0)

        # output is 0 to self.nchannel
        self.nchannel = self.L + self.start_channel - 1

        # Number of frames
        self.nframe = self.wf_out.getnframes()

        # Flag for stop stream
        self.flg_stop = 0

        # Format
        if self.format == pyaudio.paInt16:
            self.format_np = np.int16
            self.nbyte = 2
        elif self.format == pyaudio.paInt32:
            self.format_np = np.int32
            self.nbyte = 4
        elif self.format == pyaudio.paInt8:
            self.format_np = np.int8
            self.nbyte = 1
        elif self.format == pyaudio.paUInt8:
            self.format_np = np.uint8
            self.nbyte = 1
        elif self.format == pyaudio.paFloat32:
            self.format_np = np.float32
            self.nbyte = 4
        else:
            print("Invalid format")
            sys.exit(0)
            return

        print("- Sampling frequency [Hz]: %d" % self.Fs)
        print("- Number of output channels: %d" % self.L)

        # Audio device information
        self.pa = pyaudio.PyAudio() #initialize pyaudio
        if self.dev_id>=0:
            out_dev_info = self.pa.get_device_info_by_index(self.dev_id)
        else: #default audio device
            out_dev_info = self.pa.get_default_output_device_info()

        print("- Device (Output): %s, SampleRate: %dHz, MaxOutputChannels: %d" % (out_dev_info['name'],int(out_dev_info['defaultSampleRate']),int(out_dev_info['maxOutputChannels'])))

        # Check audio device support
        if self.pa.is_format_supported(rate=self.Fs, output_device=out_dev_info['index'], output_channels=self.n_out_channel, output_format=self.format) == False:
            print("Error: audio driver does not support current setting")
            return None

        self.ifrm = 0
        self.pa_indata = []
        #self.playbuff = np.zeros((self.nchannel,self.chunk), dtype=self.format_np)

        # Open stream
        if self.dev_id<0:
            self.stream = self.pa.open(format=self.format,
                                       channels=self.nchannel,
                                       rate=self.Fs,
                                       input=False,
                                       output=True,
                                       frames_per_buffer=self.chunk,
                                       stream_callback=self.callback)
        else:
            self.stream = self.pa.open(format=self.format,
                                       channels=self.nchannel,
                                       rate=self.Fs,
                                       input=False,
                                       output=True,
                                       input_device_index=self.dev_id,
                                       output_device_index=self.dev_id,
                                       frames_per_buffer=self.chunk,
                                       stream_callback=self.callback)


    def start(self):
        self.ifrm = 0
        self.filt = self.exploit_transfer_func_T()
        self.stream.start_stream()
        return 0

    def terminate(self):
        self.stream.close()
        self.wf_out.close()
        self.pa.terminate()

    def callback(self, in_data, frame_count, time_info, status):
        start = time.time()

        playbuff = np.zeros((self.nchannel,self.chunk), dtype=self.format_np)
        p_data = self.wf_out.readframes(self.chunk)
        prev_nframes = int(len(p_data)/self.n_out_channel/self.nbyte)
        #start = time.time()
        data = self.convolve_filter(np.frombuffer(p_data, dtype=self.format_np).reshape(prev_nframes, self.n_out_channel).T)
        #elapsed_time = time.time()-start
        #print("- elapsed_time of convolve:{0}".format(elapsed_time) + "[sec]")
        cur_nframes = int(min(self.chunk, data.shape[1]))
        #start = time.time()
        playbuff[self.start_channel-1:self.nchannel,0:cur_nframes] = self.float2int(data[:,0:cur_nframes])
        #elapsed_time = time.time()-start
        #print("- elapsed_time of cast:{0}".format(elapsed_time) + "[sec]")
        pa_outdata = (playbuff.T).reshape((self.chunk*self.nchannel,1))
        
        '''
        playbuff = np.zeros((self.nchannel,self.chunk), dtype=self.format_np)
        data = self.wf_out.readframes(self.chunk)
        cur_nframes = int(len(data)/self.n_out_channel/self.nbyte)
        playbuff[self.start_channel-1:self.nchannel,0:cur_nframes] = np.frombuffer(data, dtype=self.format_np).reshape(cur_nframes, self.n_out_channel).T
        pa_outdata = (playbuff.T).reshape((self.chunk*self.nchannel,1))
        '''
        
        self.ifrm += 1
        if self.ifrm == int(np.ceil(self.nframe/self.chunk)):
            self.wf_out.rewind()
            self.ifrm = 0
        elapsed_time = time.time()-start
        print("- elapsed_time of filter:{0}".format(elapsed_time) + "[sec]")
        return (pa_outdata, pyaudio.paContinue)

    def waitstream(self):
        #while self.flg_stop<1:
        while True:
            time.sleep(0.5)
            #if self.stream.is_active()==0:
                #self.flg_stop = 1

    def update_source(self):
        while True:
            print('x y z = ?\n')
            r_s =  list(map(float, input().split()))
            if len(r_s) != 3:
                print('input should be 3 float numbers separated with space.')
                continue
            self.r_s = np.array(r_s,dtype=np.float)
            self.lock.acquire()
            self.filt = self.exploit_transfer_func_T()
            self.lock.release()

    def float2int(self,data):
        if data.dtype == 'float32' or data.dtype == 'float64':
            max_y = np.max(np.abs(data))
        elif data.dtype == 'uint8':
            data = data - 128
            max_y = 128
        elif data.dtype == 'int16':
            max_y = np.abs(np.iinfo(np.int16).min)
        else:
            max_y = np.abs(np.iinfo(np.int16).min)
        max_y *= 8
        if max_y == 0.0:
            max_y = 1.0
        return self.format_np(data / max_y * np.abs(np.iinfo(self.format_np).min))

    def convolve_filter(self,data):
        '''
        input(data):(1,*), * is undetermined.
        output: array(L,*), ** is undetermined. * == ** is not necessarily true.
        '''
        rate = 2*self.f_max
        data = librosa.resample(data.astype('float32'),self.Fs,rate)

        if data.ndim == 2:
            length = data.shape[1]
            data = data[0,:]
        elif data.ndim == 1:
            length = data.shape[0]
        else:
            print('invalid wave file!!!')
            sys.exit(0)

        _ans = np.zeros((self.L,length))

        self.lock.acquire()
        _ans = signal.fftconvolve(data,self.filt,mode='same', axes=1)[0:length]
        '''
        for i in range(self.L):
            _ans[i,:] =  np.convolve(data,self.filt[i,:],mode='same')[0:length]
        '''
        self.lock.release()

        test = librosa.resample(_ans[0,:],rate,self.Fs)
        resampled = np.zeros((self.L,test.shape[0]))
        for i in range(self.L):
            resampled[i,:] = librosa.resample(_ans[i,:],rate,self.Fs)
        return resampled
        
        
    def exploit_transfer_func_T(self):
        '''
        M: the even number of omega.
        return filter.
        '''
        #t = 0.5
        omega_s = np.linspace(0,self.omega_mx,self.M+1)
        d_s = np.zeros((self.L,2*self.M), dtype=np.complex)
        for i, omega in enumerate(omega_s):
            if omega == 0:
                continue
            elif omega < self.omega_low:
                continue
            k = omega/self.c
            d = self.__exploit_d(k=k).flatten()
            #d = np.exp(-1j*omega*t)*np.ones(self.L)
            d_s[:,self.M-1+i] = d
            if i != self.M:
                d_s[:,self.M-1-i] = np.conj(d)
        d_s = np.append(d_s,d_s[:,0:self.M-1],axis=1)[:,self.M-1:]
        ans = spfft.ifft(d_s,axis=1)
        ans = np.real(np.append(ans,ans[:,0:self.M],axis=1)[:,self.M:])
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

if __name__== '__main__':
    #pap = paplay("./test/tsp_out.wav",1)
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


    obj = RealtimeSynthesizer("./tests/asano.wav",r, r_c, r_s, Rint, gamma, is_silent,f_max=1000,f_low=200,M=256,start_channel=5,Fs=0,dev_id=2)

    w_th1 = threading.Thread(target=obj.waitstream)
    #w_th2 = threading.Thread(target=obj.update_source)

    obj.start()
    
    w_th1.start()
    #w_th2.start()

    w_th1.join()

    obj.terminate()
