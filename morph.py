import numpy as np
import matplotlib.pyplot as plt
'''
Opening and Closing Pattern Spectrum implementation
'''


def dilation(signal, n):
    '''
    signal: input time series
    n: structure element = window size
    '''
    lens = signal.shape[0]
    new_signal = np.zeros(lens)

    #### front #########
    for i in range(n):
        tmp_signal = signal[:n+i]
        new_signal[i] = np.max(tmp_signal)

    #### middle #########
    for i in range (n,lens-n,1):
        tmp_signal = signal[i-n:i+n]
        new_signal[i] = np.max(tmp_signal)
    
    #### end #########
    for i in range(n):
        tmp_signal = signal[lens-n-i:]
        new_signal[lens-1-i] = np.max(tmp_signal)
    return new_signal

def erosion(signal, n):
    '''
    signal: input time series
    n: structure element = window size
    '''
    lens = signal.shape[0]
    new_signal = np.zeros(lens)

    #### front #########
    for i in range(n):
        tmp_signal = signal[:n+i]
        new_signal[i] = np.min(tmp_signal)

    #### middle #########
    for i in range (n,lens-n,1):
        tmp_signal = signal[i-n:i+n]
        new_signal[i] = np.min(tmp_signal)
    
    #### end #########
    for i in range(n):
        tmp_signal = signal[lens-n-i:]
        new_signal[lens-1-i] = np.min(tmp_signal)
    return new_signal

def opening(signal, n):
    '''
    dilation then erosion
    '''
    signal = dilation(signal, n)
    signal = erosion(signal, n)
    return signal


def closing(signal, n):
    '''
    erosion then dilation
    '''    
    signal = erosion(signal, n)
    signal = dilation(signal, n)
    return signal



def open_pattern_spectrum(signal, n, scale):

    '''
    calculate the change of the area between two opening opertions
    signal: input time series
    n: structure element
    scale: # of the opening operation
    '''
    ## non negative
    print('input signal', signal)
    signal = signal + np.abs(np.min(signal))
    print('singal non negative', signal)
    ###
    for i in range(scale):
        signal_2 = signal 
        print('i=',i)
        signal = opening(signal, n)
        print('signal=',signal)

    diff = signal - signal_2 
    print('diff',diff)
    area = np.abs(np.sum(diff))
    print('area',area)
    return area

def close_pattern_spectrum(signal, n, scale):
    '''
    calculate the change of the area between two opening opertions
    signal: input time series
    n: structure element
    scale: # of the closing operation
    '''    
    signal = signal + np.abs(np.min(signal))

    for i in range(scale):
        signal_2 = signal 
        signal = closing(signal, n)
    diff = signal_2 - signal
    area = np.abs(np.sum(diff))

    return area
#########################
def curve_length(signal):
    '''
    calculate signal curve length
    '''
    tmp = 0
    for i in range (signal.shape[0]-1):
        tmp += np.abs(signal[i+1] - signal[i])
    return tmp/(signal.shape[0]-1)



def num_of_peaks(signal):
    '''
    calculate # of peak
    '''
    tmp = 0
    for i in range(signal.shape[0]-2):
        tmp2 = np.sign(signal[i+2]-signal[i+1]) - np.sign(signal[i+1]-signal[i])
        tmp += np.max([0,tmp2])
    return int(tmp/2)


def avg_nonlinear_energy(signal):
    tmp = 0
    for i in range (signal.shape[0]-2):
        tmp += signal[i+1]*signal[i+1] - signal[i]*signal[i+2]
    return tmp/(signal.shape[0]-2)


##########################################################
'''
a = np.array([1,12,3,14,5,16,7,16,5,14,4,15,16,7,18,7,6])
cura = curve_length(a)
peak = num_of_peaks(a)
egya = avg_nonlinear_energy(a)
print('cura a',cura)
print('peak a',peak)
print('egya a',egya)

b = np.array([2,2,2,4,4,4,5,5,5,5,5,5,5,5,3,3,3])
curb = curve_length(b)
peakb = num_of_peaks(b)
egyb = avg_nonlinear_energy(b)

print('curb b',curb)
print('peak b',peakb)
print('egyb b',egyb)

##########################

plt.plot(a)
plt.plot(d)
plt.plot(e)

plt.show()
plt.figure()
plt.plot(d2)
plt.plot(e2)
plt.shorw()
'''