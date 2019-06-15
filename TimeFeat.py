import numpy as np

def mean_power(x):
	return np.mean(np.square(x))
def mean(x):
	return np.mean(np.abs(x))
def std(x):
	return np.std(x)
def first_diff(x):
	return np.mean(np.abs(np.diff(x)))
def second_diff(x):
	order = 2
	if(order == 0):
		array = x
	else:
		array = x[order:] - x[0:-order]
	return np.mean(np.abs(array))

def hoc(x, order):
	#order = 10
	#HOC_list = []
	#for i in range(order):
	if(order == 0):
		array = x
	else:
		array = x[order:] - x[0:-order]
	zcr = ((array[:-1] * array[1:])<0).sum() / len(array[1:])
	#HOC_list.append(zcr)
	return zcr
	'''
	a = np.array([1,3,5,7,9,11,13,15,17])
	print(np.diff(a))
	print(a[1:] - a[0:-1])


	print(a[2:] - a[0:-2])

	print(a[3:] - a[0:-3])
	'''

def nsi(x):
	mean_list = []
	scale = 20
	lenght = len(x)
	for i in range(int(lenght / scale)):
		local = x[0:scale]
		mean_list.append(np.mean(local))
		x = x[scale:]

	return np.std(mean_list)



def hjcomp(X, D=None):
    def _mob(x):
    	return np.sqrt(np.var(np.diff(x))/np.var(x))
    return _mob(np.diff(X))/_mob(X)


#!/usr/bin/python3


import os
import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer


def interval_t(size,num_val=50,kmax=None):
    ### Generate sequence of interval times, k
    if kmax is None:
        k_stop = size//2
    else:
        k_stop = kmax
    if k_stop > size//2:## prohibit going larger than N/2
        k_stop = size//2
        print("Warning: k cannot be longer than N/2")
        
    k = np.logspace(start=np.log2(2),stop=np.log2(k_stop),base=2,num=num_val,dtype=np.int)
    return np.unique(k);

def init_lib():
    libdir = os.path.dirname(__file__)
    libfile = os.path.join(libdir, "libhfd.so")
    lib = ctypes.CDLL(libfile)

    rwptr = ndpointer(float, flags=('C','A','W'))
    rwptr_sizet = ndpointer(ctypes.c_size_t, flags=('C','A','W'))

    lib.curve_length.restype = ctypes.c_int
    lib.curve_length.argtypes = [rwptr_sizet, ctypes.c_size_t, rwptr, ctypes.c_size_t, rwptr]

    return lib;
if __name__ == '__main__':
	print(HOC(1))
