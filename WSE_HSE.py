from scipy.io import loadmat
from os.path import basename
import numpy as np
import pickle
import feature_extract as f
import utils as f
from pywt import wavedec
from math import log
import GrayCode
from time import clock



def get_sequency_list(inputArray):
    """ Sort input 1D array into sequency order
    Utilizes gray code generation from a Python recipe from Internet.
    """
    length = inputArray.size
    bitlength = int(log(length,2))
    # Gray Code
    graycodes=GrayCode.GrayCode(bitlength)
    # Bitreverse of gray code
    bitreverse = [int(graycodes[i][::-1],2) for i in range(length)]
    
    outputArray = inputArray.copy()
    outputArray[bitreverse] = inputArray[:]

    return outputArray



def SFWHT(X):
    """ 'Slower' Fast Walsh-Hadamard Transform
    Step#1 Get sequency-ordered input
    Step#2 Perform Hadamard Transform
    """
    # if you dont know, take the input x = [1:8] and you would know 
    # when m = 3 , this fiunction would do three matrix multiplication task 
    # x=get_sequency_list(X)
    x= np.array(X)
    #print('get_sequency_list',x)
    M = int(log(x.size,2))
    x = x[:(2**M)]
    #print('x size',x.size)

    N = x.size
    out = x.copy()
    for m in range(M):
        #print('m',m)
        outtemp = out.copy()         
        step = 2**m
        #print('step',step)
        numCalc = 2**m
        #print('numcalc',numCalc)
        for g in range(0,N,2*step): # number of groups
            #print('g',g)

            for c in range(numCalc):
                #print('c',c)
                index = g + c
                out[index] = outtemp[index] + outtemp[index+step]
                out[index+step] = outtemp[index] - outtemp[index+step]
               # print('out',out)
    #print ('result:',out/float(N))
    return out/float(N)

# to do:
# first call the wavelet (walsh) transform function
# then call the sample entropy function  
# then use the small script to test the result 
def shannon_ent(time_series):
    """Return the Shannon Entropy of the sample data.
    Args:
        time_series: Vector or string of the sample data
    Returns:
        The Shannon Entropy as float value
    """

    # Check if string
    if not isinstance(time_series, str):
        time_series = list(time_series)

    # Create a frequency data
    data_set = list(set(time_series))
    #print('dataset',data_set)
    freq_list = []
    for entry in data_set:
        counter = 0.
        for i in time_series:
            if i == entry:
                counter += 1
        freq_list.append(float(counter) / len(time_series))

    # Shannon entropy
    ent = 0.0
    for freq in freq_list:
       # print('freq',freq)
        ent += freq * np.log2(freq)
    ent = -ent

    return ent
def WSE(x):

    """
    Compute Walsh spectral entropy
    Args:
        time_series: Vector of the sample data
    Returns:
        Walsh spectral entropy
    """
    x = np.array(x)
    x = SFWHT(x)
    sum = 0
    ent = 0
    print(x)
    for i in range(x.shape[0]):
        x[i]=x[i]*x[i]
        sum +=x[i]
    print('squarex',x)
    for i in range(x.shape[0]):
        x[i]=x[i]/sum
    print('normalize x',x)


    for i in range(x.shape[0]):
        ent +=x[i]*np.log2(x[i])
    ent = ent/np.log2(x.shape[0])
    ent = -ent
    print('en',ent)
    return ent

def HSE(x):
    """
    Compute Haar spectral entropy
    Args:
        time_series: Vector of the sample data
    Returns:
        Haar spectral entropy
    """
    x = np.array(x)
    M = int(log(x.size,2))
    sum = 0
    ent = 0
    coeffs = wavedec(x ,'db4', level=M)
    #print(coeffs)
    x =np.array([])
    for i in range(len(coeffs)):
        #print('coeffs[i]',coeffs[i])
        x = np.concatenate((x, coeffs[i]), axis=None)

    print('x',x)
    for i in range(x.shape[0]):
        x[i]=x[i]*x[i]
        sum +=x[i]
    print('squarex',x)
    for i in range(x.shape[0]):
        x[i]=x[i]/sum
    print('normalize x',x)


    for i in range(x.shape[0]):
        ent +=x[i]*np.log2(x[i])
    ent = ent/np.log2(x.shape[0])
    ent = -ent
    
    print('en',ent)
    return ent
  

   
if __name__ == "__main__":
    x = np.random.random(1024**2)
    x = [1,1,1,0,0,0,1,1,1,0,1,1,1,1,1]
    #dic = f.pickle_op(file_name = 'saved_pickle/whole_trial_26_eog.p', mode = 'r')
    # dic = f.pickle_op(file_name = 'saved_pickle/{}_trial_{}_{}.p'.format(args.input_type, args.sub, args.dir), mode = 'r')
    '''if(args.dir == 'no'):
        print('Without artifact removal')
    elif(args.dir == 'RSTM'):
        print('Use RSTM to remove eye artifact')
    elif(args.dir == 'eog'):
        print('Use EOG to remove eye artifact')
    '''
   # y = dic['y'] # 26 x 27
   # x = dic['x'] # 26 x 27 x 30 x 6080 or 26 x 513 x 30 x 419
   # x = f.sub_baseline(x)
   # feat , featlog = f.feature_extract(x,wh.WSE, 
   # 'WSE', 'whole_feature_pickle/26eog/WSE.p', type = 'new')
    x = np.array(x)
    ans = WSE(x)
    #y = WSE(x)
    #ans = HSE(x[0][0][0])
    #print('WSE', y)
    print ('HSE',ans)
