import numpy as np
#import pyeeg
#import biosppyMe.signals as bios
from numpy import abs, sum, linspace
from numpy.fft import rfft
from scipy import stats
import scipy.signal as signal
#import pyeeg
import pywt
import matplotlib.pyplot as plt
import TimeFeature as tf
'''
def psd(channel, window, overlap, start_freq, end_freq, Fs = 200):
	duration = len(channel) / Fs
	samples = (duration - window) / (window - overlap) + 1
	offset = 0
	output = []
	for i in range(int(samples)):
		compute = channel[0 + offset : window * Fs + offset]
		offset = offset + (window - overlap) * Fs
		f, Pxx = signal.periodogram(compute, Fs) # Estimate power spectral density
		start_idx = np.argmin(np.abs(f - start_freq))
		end_idx   = np.argmin(np.abs(f - end_freq))
		psdensity = np.mean(Pxx[start_idx:end_idx])
		output.append(psdensity)
	output = np.array(output)
	return output
'''
def psd(x, start_freq, end_freq, Fs = 200.0):
    # freq resolution (interval of f ): sampling freq / number of sample points
    # impossible to measure freq above the Nyquist limit(Fs/2), 
    # get rid of above Nyquist limit signal, and double the magnitude by 2
    # to make the total amplitude equal to one, need to normalize by number of sample
    f, Pxx = signal.periodogram(x, Fs) # Estimate power spectral density
    Pxx = np.log(Pxx + 1e-6)
    Pxx = Pxx / (np.sum(Pxx) + 1e-6)
    start_idx = np.argmin(np.abs(f - start_freq))
    end_idx   = np.argmin(np.abs(f - end_freq))
    psdensity = np.mean(Pxx[start_idx:end_idx])
    return psdensity
def psdMAV(x):
    # freq resolution (interval of f ): sampling freq / number of sample points
    # impossible to measure freq above the Nyquist limit(Fs/2), 
    # get rid of above Nyquist limit signal, and double the magnitude by 2
    # to make the total amplitude equal to one, need to normalize by number of sample
    theta = [7,10]
    alpha = [11,19]
    beta  = [20,29]
    gamma = [30,45]

    f, Pxx = signal.periodogram(x, 200) # Estimate power spectral density
    Pxx = np.log(Pxx + 1e-6)
    Pxx = Pxx / (np.sum(Pxx) + 1e-6)
    start_freq1, end_freq1 = 11, 19
    start_freq2, end_freq2 = 30, 45
    start_idx = np.argmin(np.abs(f - start_freq1))
    end_idx   = np.argmin(np.abs(f - end_freq1))
    psdensity1 = np.mean(Pxx[start_idx:end_idx])

    start_idx = np.argmin(np.abs(f - start_freq2))
    end_idx   = np.argmin(np.abs(f - end_freq2))
    psdensity2 = np.mean(Pxx[start_idx:end_idx])

    return psdensity2 / psdensity1
def stft_power(x, start_freq, end_freq, Fs = 200.0):
    f, t, Zxx = signal.stft(x, fs = Fs, window = 'hann', nperseg = 256, noverlap = 0, nfft = None,
                detrend = False, return_onesided = True, boundary = 'zeros', padded = True, axis = -1)

    s_idx = np.argmin(np.abs(f - start_freq))
    e_idx = np.argmin(np.abs(f - end_freq))

    Zxx = np.abs(Zxx)
    Zxx = np.log(Zxx + 1e-6)

    Zxx = Zxx / (np.sum(Zxx) + 1e-6)



    return np.mean(Zxx[s_idx:e_idx])

def waveletF(x):
    '''
        detail(h)     : from the high pass filter, output high freq part
        approximate(g): from the low  pass filter, output low  freq part -> can continue

        The signal have 32 sample,
        -------------------------------
        |Level|  Frequency  | Samples |
        -------------------------------
        |  3  |0    to fn/8 |    4    |
        |     |fn/8 to fn/4 |    4    |
        -------------------------------
        |  2  |fn/4 to fn/2 |    8    |
        -------------------------------
        |  1  |fn/2 to fn   |    16   |
        -------------------------------
    '''

    #w = pywt.wavelist()[wt]
    Decomp_sig = pywt.wavedec(x, 'sym17', level=4)
    cA4 = Decomp_sig[0]# 3.125~6.25
    theta = psd(cA4, 4, 7, Fs = 200.0)
    cD4 = Decomp_sig[1]# 6.25~12.5
    alpha = psd(cD4, 8, 13, Fs = 200.0)
    cD3 = Decomp_sig[2]# 12.25~25
    beta = psd(x, 16, 25, Fs = 200.0)
    cD2 = Decomp_sig[3]# 25~50
    gamma = psd(x, 30, 45, Fs = 200.0)
    cD1 = Decomp_sig[4]# 50~100

    '''
    Decomp_sig = Decomp_sig[0:-1]
    feature = []
    for i in range(len(Decomp_sig)):
        MAV = np.mean(np.abs(Decomp_sig[i]))
        #AP = np.mean(np.square(Decomp_sig[i]))
        SD = np.std(Decomp_sig[i])
        #feature.append(MAV)
        #feature.append(AP)
        #feature.append(SD)
    for i in range(len(Decomp_sig)-1):
        RAM = np.mean(np.abs(Decomp_sig[i])) / np.mean(np.abs(Decomp_sig[i + 1]))
        feature.append(RAM)
    '''
    return theta

def waveletT(x):
    '''
        detail(h)     : from the high pass filter, output high freq part
        approximate(g): from the low  pass filter, output low  freq part -> can continue

        The signal have 32 sample,
        -------------------------------
        |Level|  Frequency  | Samples |
        -------------------------------
        |  3  |0    to fn/8 |    4    |
        |     |fn/8 to fn/4 |    4    |
        -------------------------------
        |  2  |fn/4 to fn/2 |    8    |
        -------------------------------
        |  1  |fn/2 to fn   |    16   |
        -------------------------------
    '''
    #w = pywt.wavelist()[wt]
    Decomp_sig = pywt.wavedec(x, 'sym17', level=4)

    cA4 = Decomp_sig[0]# 3.125~6.25
    cD4 = Decomp_sig[1]# 6.25~12.5
    cD3 = Decomp_sig[2]# 12.25~25
    cD2 = Decomp_sig[3]# 25~50
    cD1 = Decomp_sig[4]# 50~100

    #a = pywt.idwt( cA = None, cD = Decomp_sig[3], wavelet = 'sym17')
    #f, Pxx = signal.periodogram(a, 200)
    #plt.semilogy(f, Pxx)
    #plt.show()
    #print(a)
    Decomp_sig = Decomp_sig[0:-1]
    feature = []
    #for i in range(len(Decomp_sig)):
    #    MAV = np.mean(np.abs(Decomp_sig[i]))
    #    AP = np.mean(np.square(Decomp_sig[i]))
    #    SD = np.std(Decomp_sig[i])
    #    first_diff = np.mean(np.abs(np.diff(Decomp_sig[i])))
    #    second_diff = np.mean(np.abs(np.array(Decomp_sig)[i][2:] - np.array(Decomp_sig)[i][0:-2]))
#
    #    #feature.append(MAV)
    #    #feature.append(AP)
    #    #feature.append(SD)
    #    #feature.append(first_diff)
    #    #feature.append(second_diff)
    for i in range(len(Decomp_sig)-1):
        RAM = np.mean(np.abs(Decomp_sig[i])) / np.mean(np.abs(Decomp_sig[i + 1]))
        feature.append(RAM)

    return np.array(feature)
#########################
def waveletT_RAM(x):
    '''
        detail(h)     : from the high pass filter, output high freq part
        approximate(g): from the low  pass filter, output low  freq part -> can continue

        The signal have 32 sample,
        -------------------------------
        |Level|  Frequency  | Samples |
        -------------------------------
        |  3  |0    to fn/8 |    4    |
        |     |fn/8 to fn/4 |    4    |
        -------------------------------
        |  2  |fn/4 to fn/2 |    8    |
        -------------------------------
        |  1  |fn/2 to fn   |    16   |
        -------------------------------
    '''
    #w = pywt.wavelist()[wt]
    Decomp_sig = pywt.wavedec(x, 'sym17', level=4)

    cA4 = Decomp_sig[0]# 3.125~6.25
    cD4 = Decomp_sig[1]# 6.25~12.5
    cD3 = Decomp_sig[2]# 12.25~25
    cD2 = Decomp_sig[3]# 25~50
    cD1 = Decomp_sig[4]# 50~100

    #a = pywt.idwt( cA = None, cD = Decomp_sig[3], wavelet = 'sym17')
    #f, Pxx = signal.periodogram(a, 200)
    #plt.semilogy(f, Pxx)
    #plt.show()
    #print(a)
    Decomp_sig = Decomp_sig[0:-1]
    feature = []
    #for i in range(len(Decomp_sig)):
    #    MAV = np.mean(np.abs(Decomp_sig[i]))
    #    AP = np.mean(np.square(Decomp_sig[i]))
    #    SD = np.std(Decomp_sig[i])
    #    first_diff = np.mean(np.abs(np.diff(Decomp_sig[i])))
    #    second_diff = np.mean(np.abs(np.array(Decomp_sig)[i][2:] - np.array(Decomp_sig)[i][0:-2]))
#
    #    #feature.append(MAV)
    #    #feature.append(AP)
    #    #feature.append(SD)
    #    #feature.append(first_diff)
    #    #feature.append(second_diff)
    for i in range(len(Decomp_sig)-1):
        RAM = np.mean(np.abs(Decomp_sig[i])) / np.mean(np.abs(Decomp_sig[i + 1]))
        feature.append(RAM)

    return np.array(feature)
############################
def waveletT_MAV(x):
    '''
        detail(h)     : from the high pass filter, output high freq part
        approximate(g): from the low  pass filter, output low  freq part -> can continue

        The signal have 32 sample,
        -------------------------------
        |Level|  Frequency  | Samples |
        -------------------------------
        |  3  |0    to fn/8 |    4    |
        |     |fn/8 to fn/4 |    4    |
        -------------------------------
        |  2  |fn/4 to fn/2 |    8    |
        -------------------------------
        |  1  |fn/2 to fn   |    16   |
        -------------------------------
    '''
    #w = pywt.wavelist()[wt]
    Decomp_sig = pywt.wavedec(x, 'sym17', level=4)

    cA4 = Decomp_sig[0]# 3.125~6.25
    cD4 = Decomp_sig[1]# 6.25~12.5
    cD3 = Decomp_sig[2]# 12.25~25
    cD2 = Decomp_sig[3]# 25~50
    cD1 = Decomp_sig[4]# 50~100

    #a = pywt.idwt( cA = None, cD = Decomp_sig[3], wavelet = 'sym17')
    #f, Pxx = signal.periodogram(a, 200)
    #plt.semilogy(f, Pxx)
    #plt.show()
    #print(a)
    Decomp_sig = Decomp_sig[0:-1]
    feature = []
    for i in range(len(Decomp_sig)):
        MAV = np.mean(np.abs(Decomp_sig[i]))
    #    AP = np.mean(np.square(Decomp_sig[i]))
    #    SD = np.std(Decomp_sig[i])
    #    first_diff = np.mean(np.abs(np.diff(Decomp_sig[i])))
    #    second_diff = np.mean(np.abs(np.array(Decomp_sig)[i][2:] - np.array(Decomp_sig)[i][0:-2]))
#
        feature.append(MAV)
    #    #feature.append(AP)
    #    #feature.append(SD)
    #    #feature.append(first_diff)
    #    #feature.append(second_diff)
    #for i in range(len(Decomp_sig)-1):
    #   RAM = np.mean(np.abs(Decomp_sig[i])) / np.mean(np.abs(Decomp_sig[i + 1]))
    #   feature.append(RAM)

    return np.array(feature)
#############################
def waveletT_AP(x):
    '''
        detail(h)     : from the high pass filter, output high freq part
        approximate(g): from the low  pass filter, output low  freq part -> can continue

        The signal have 32 sample,
        -------------------------------
        |Level|  Frequency  | Samples |
        -------------------------------
        |  3  |0    to fn/8 |    4    |
        |     |fn/8 to fn/4 |    4    |
        -------------------------------
        |  2  |fn/4 to fn/2 |    8    |
        -------------------------------
        |  1  |fn/2 to fn   |    16   |
        -------------------------------
    '''
    #w = pywt.wavelist()[wt]
    Decomp_sig = pywt.wavedec(x, 'sym17', level=4)

    cA4 = Decomp_sig[0]# 3.125~6.25
    cD4 = Decomp_sig[1]# 6.25~12.5
    cD3 = Decomp_sig[2]# 12.25~25
    cD2 = Decomp_sig[3]# 25~50
    cD1 = Decomp_sig[4]# 50~100

    #a = pywt.idwt( cA = None, cD = Decomp_sig[3], wavelet = 'sym17')
    #f, Pxx = signal.periodogram(a, 200)
    #plt.semilogy(f, Pxx)
    #plt.show()
    #print(a)
    Decomp_sig = Decomp_sig[0:-1]
    feature = []
    for i in range(len(Decomp_sig)):
    #    MAV = np.mean(np.abs(Decomp_sig[i]))
        AP = np.mean(np.square(Decomp_sig[i]))
    #    SD = np.std(Decomp_sig[i])
    #    first_diff = np.mean(np.abs(np.diff(Decomp_sig[i])))
    #    second_diff = np.mean(np.abs(np.array(Decomp_sig)[i][2:] - np.array(Decomp_sig)[i][0:-2]))
#
    #    #feature.append(MAV)
        feature.append(AP)
    #    #feature.append(SD)
    #    #feature.append(first_diff)
    #    #feature.append(second_diff)
    #for i in range(len(Decomp_sig)-1):
    #    RAM = np.mean(np.abs(Decomp_sig[i])) / np.mean(np.abs(Decomp_sig[i + 1]))
    #    feature.append(RAM)

    return np.array(feature)
##############################
def waveletT_SD(x):
    '''
        detail(h)     : from the high pass filter, output high freq part
        approximate(g): from the low  pass filter, output low  freq part -> can continue

        The signal have 32 sample,
        -------------------------------
        |Level|  Frequency  | Samples |
        -------------------------------
        |  3  |0    to fn/8 |    4    |
        |     |fn/8 to fn/4 |    4    |
        -------------------------------
        |  2  |fn/4 to fn/2 |    8    |
        -------------------------------
        |  1  |fn/2 to fn   |    16   |
        -------------------------------
    '''
    #w = pywt.wavelist()[wt]
    Decomp_sig = pywt.wavedec(x, 'sym17', level=4)

    cA4 = Decomp_sig[0]# 3.125~6.25
    cD4 = Decomp_sig[1]# 6.25~12.5
    cD3 = Decomp_sig[2]# 12.25~25
    cD2 = Decomp_sig[3]# 25~50
    cD1 = Decomp_sig[4]# 50~100

    #a = pywt.idwt( cA = None, cD = Decomp_sig[3], wavelet = 'sym17')
    #f, Pxx = signal.periodogram(a, 200)
    #plt.semilogy(f, Pxx)
    #plt.show()
    #print(a)
    Decomp_sig = Decomp_sig[0:-1]
    feature = []
    for i in range(len(Decomp_sig)):
    #    MAV = np.mean(np.abs(Decomp_sig[i]))
    #    AP = np.mean(np.square(Decomp_sig[i]))
        SD = np.std(Decomp_sig[i])
    #    first_diff = np.mean(np.abs(np.diff(Decomp_sig[i])))
    #    second_diff = np.mean(np.abs(np.array(Decomp_sig)[i][2:] - np.array(Decomp_sig)[i][0:-2]))
#
    #    #feature.append(MAV)
    #    #feature.append(AP)
        feature.append(SD)
    #    #feature.append(first_diff)
    #    #feature.append(second_diff)
    #for i in range(len(Decomp_sig)-1):
    #    RAM = np.mean(np.abs(Decomp_sig[i])) / np.mean(np.abs(Decomp_sig[i + 1]))
    #   feature.append(RAM)

    return np.array(feature)
#########################

def waveletT_first_diff(x):
    '''
        detail(h)     : from the high pass filter, output high freq part
        approximate(g): from the low  pass filter, output low  freq part -> can continue

        The signal have 32 sample,
        -------------------------------
        |Level|  Frequency  | Samples |
        -------------------------------
        |  3  |0    to fn/8 |    4    |
        |     |fn/8 to fn/4 |    4    |
        -------------------------------
        |  2  |fn/4 to fn/2 |    8    |
        -------------------------------
        |  1  |fn/2 to fn   |    16   |
        -------------------------------
    '''
    #w = pywt.wavelist()[wt]
    Decomp_sig = pywt.wavedec(x, 'sym17', level=4)

    cA4 = Decomp_sig[0]# 3.125~6.25
    cD4 = Decomp_sig[1]# 6.25~12.5
    cD3 = Decomp_sig[2]# 12.25~25
    cD2 = Decomp_sig[3]# 25~50
    cD1 = Decomp_sig[4]# 50~100

    #a = pywt.idwt( cA = None, cD = Decomp_sig[3], wavelet = 'sym17')
    #f, Pxx = signal.periodogram(a, 200)
    #plt.semilogy(f, Pxx)
    #plt.show()
    #print(a)
    Decomp_sig = Decomp_sig[0:-1]
    feature = []
    for i in range(len(Decomp_sig)):
    #    MAV = np.mean(np.abs(Decomp_sig[i]))
    #    AP = np.mean(np.square(Decomp_sig[i]))
    #    SD = np.std(Decomp_sig[i])
        first_diff = np.mean(np.abs(np.diff(Decomp_sig[i])))
    #    second_diff = np.mean(np.abs(np.array(Decomp_sig)[i][2:] - np.array(Decomp_sig)[i][0:-2]))
#
    #    #feature.append(MAV)
    #    #feature.append(AP)
        #feature.append(SD)
        feature.append(first_diff)
    #    #feature.append(second_diff)
    #for i in range(len(Decomp_sig)-1):
    #    RAM = np.mean(np.abs(Decomp_sig[i])) / np.mean(np.abs(Decomp_sig[i + 1]))
    #   feature.append(RAM)

    return np.array(feature)
#########################

def waveletT_second_diff(x):
    '''
        detail(h)     : from the high pass filter, output high freq part
        approximate(g): from the low  pass filter, output low  freq part -> can continue

        The signal have 32 sample,
        -------------------------------
        |Level|  Frequency  | Samples |
        -------------------------------
        |  3  |0    to fn/8 |    4    |
        |     |fn/8 to fn/4 |    4    |
        -------------------------------
        |  2  |fn/4 to fn/2 |    8    |
        -------------------------------
        |  1  |fn/2 to fn   |    16   |
        -------------------------------
    '''
    #w = pywt.wavelist()[wt]
    Decomp_sig = pywt.wavedec(x, 'sym17', level=4)

    cA4 = Decomp_sig[0]# 3.125~6.25
    cD4 = Decomp_sig[1]# 6.25~12.5
    cD3 = Decomp_sig[2]# 12.25~25
    cD2 = Decomp_sig[3]# 25~50
    cD1 = Decomp_sig[4]# 50~100

    #a = pywt.idwt( cA = None, cD = Decomp_sig[3], wavelet = 'sym17')
    #f, Pxx = signal.periodogram(a, 200)
    #plt.semilogy(f, Pxx)
    #plt.show()
    #print(a)
    Decomp_sig = Decomp_sig[0:-1]
    feature = []
    for i in range(len(Decomp_sig)):
    #    MAV = np.mean(np.abs(Decomp_sig[i]))
    #    AP = np.mean(np.square(Decomp_sig[i]))
    #   SD = np.std(Decomp_sig[i])
    #    first_diff = np.mean(np.abs(np.diff(Decomp_sig[i])))
        second_diff = np.mean(np.abs(np.array(Decomp_sig)[i][2:] - np.array(Decomp_sig)[i][0:-2]))
#
    #    #feature.append(MAV)
    #    #feature.append(AP)
        #feature.append(SD)
    #    #feature.append(first_diff)
        feature.append(second_diff)
    #for i in range(len(Decomp_sig)-1):
    #    RAM = np.mean(np.abs(Decomp_sig[i])) / np.mean(np.abs(Decomp_sig[i + 1]))
    #   feature.append(RAM)

    return np.array(feature)
#########################
def spectral_entropy(x, start_freq, end_freq, Fs = 200.0):
    f, Pxx = signal.periodogram(x, Fs)
    Pxx = np.square(Pxx) / len(Pxx)
    Pxx = Pxx / (np.sum(Pxx) + 1e-6)

    start_idx = np.argmin(np.abs(f - start_freq))
    end_idx   = np.argmin(np.abs(f - end_freq))
    p = Pxx[start_idx:end_idx]
    return np.sum(p*np.log(p + 1e-6)) * -1
def bin_power(X, Band, Fs):
    """Compute power in each frequency bin specified by Band from FFT result of
    X. By default, X is a real signal.
    Note
    -----
    A real signal can be synthesized, thus not real.
    Parameters
    -----------
    Band
        list
        boundary frequencies (in Hz) of bins. They can be unequal bins, e.g.
        [0.5,4,7,12,30] which are delta, theta, alpha and beta respectively.
        You can also use range() function of Python to generate equal bins and
        pass the generated list to this function.
        Each element of Band is a physical frequency and shall not exceed the
        Nyquist frequency, i.e., half of sampling frequency.
     X
        list
        a 1-D real time series.
    Fs
        integer
        the sampling rate in physical frequency
    Returns
    -------
    Power
        list
        spectral power in each frequency bin.
    Power_ratio
        list
        spectral power in each frequency bin normalized by total power in ALL
        frequency bins.
    """

    C = np.fft.fft(X)
    #C = abs(C)
    C = np.square(abs(C))
    Power = np.zeros(len(Band) - 1)
    for Freq_Index in range(0, len(Band) - 1):
        Freq = float(Band[Freq_Index])
        Next_Freq = float(Band[Freq_Index + 1])

        seg = C[int(np.floor(
                Freq / Fs * len(X)
            )): int(np.floor(Next_Freq / Fs * len(X)))]
        
        #power = sum(seg)
        power = np.sqrt(np.mean(seg))
        Power[Freq_Index] = power
        '''
        Power[Freq_Index] = np.sqrt(np.mean(
            C[int(np.floor(
                Freq / Fs * len(X)
            )): int(np.floor(Next_Freq / Fs * len(X)))]
        ))'''
    Power_Ratio = Power / sum(Power)
    return Power, Power_Ratio

if __name__ == '__main__':
	pass
