import numpy as np
#import simpleaudio as sa
import scipy.io.wavfile as sw
import wave
from wavefile import write

'''
def audioplay(fs, y):
    yout = np.iinfo(np.int16).max / np.max(np.abs(y)) * y
    yout = yout.astype(np.int16)
    play_obj = sa.play_buffer(yout, y.ndim, 2, fs)
'''

def wavread(wavefile):
    fs, y = sw.read(wavefile)
    if y.dtype == 'float32' or y.dtype == 'float64':
        max_y = 1
    elif y.dtype == 'uint8':
        y = y - 128
        max_y = 128
    elif y.dtype == 'int16':
        max_y = np.abs(np.iinfo(np.int16).min)
    else:
        max_y = np.abs(np.iinfo(np.int16).min)
    y = y / max_y
    y = y.astype(np.float32)

    return fs, y


def wavwrite(wavefile, fs, data, nchannel):
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
    data = np.int16(data / max_y * np.abs(np.iinfo(np.int16).min))
    write(wavefile, fs, data)
    '''
    w = wave.Wave_write(wavefile)
    w.setnchannels(nchannel)
    w.setsampwidth(2)
    w.setframerate(fs)
    w.writeframes(data)
    w.close()
    '''