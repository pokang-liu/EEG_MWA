import itertools
import os
import time
import numpy as np
import itertools
from math import log
###
#  RCMSE and RCMPE
##
def util_granulate_time_series(time_series, scale):
    """Extract coarse-grained time series
    Args:
        time_series: Time series
        scale: Scale factor
    Returns:
        Vector of coarse-grained time series with given scale factor
    """
    n = len(time_series)
    b = int(np.fix(n / scale))
    cts = [0] * b
    for i in range(b):
        cts[i] = np.mean(time_series[i * scale: (i + 1) * scale])
    return cts
    
def RC_composite_multiscale_entropy(time_series, sample_length, scale, m, tolerance=None):
    """Calculate the Composite Multiscale Entropy of the given time series.
    Args:
        time_series: Time series for analysis
        sample_length: Number of sequential points of the time series
        scale: Scale factor
        m: equal to sample length
        tolerance: Tolerance (default = 0.1...0.2 * std(time_series))
    Returns:
        RC Composite Multiscale Entropy
    """
    A_sum = 0
    B_sum = 0
    epsilon = 0.0000001
    for i in range(scale):
        tmp = util_granulate_time_series(time_series[i:], scale)
        A_B = RC_sample_entropy(tmp, sample_length, tolerance)
        # print(A_B)
        B_sum += A_B[m + sample_length - 1][0]
        A_sum += A_B[m - 1][0]
    rcmse = - np.log((A_sum) / B_sum)
    return rcmse


def RC_sample_entropy(time_series, sample_length, tolerance=None):
    """Calculate and return Sample Entropy of the given time series.
    Distance between two vectors defined as Euclidean distance and can
    be changed in future releases
    Args:
        time_series: Vector or string of the sample data
        sample_length (m): Number of sequential points of the time series
        tolerance: Tolerance (default = 0.1...0.2 * std(time_series))
    Returns:
        Vector containing RC Sample Entropy (float)
    """
    if tolerance is None:
        tolerance = 0.1 * np.std(time_series)

    n = len(time_series)
    prev = np.zeros(n)
    curr = np.zeros(n)
    A = np.zeros((sample_length, 1))  # number of matches for m = [1,...,template_length - 1]
    B = np.zeros((sample_length, 1))  # number of matches for m = [1,...,template_length]

    for i in range(n - 1):
        nj = n - i - 1
        ts1 = time_series[i]
        for jj in range(nj):
            j = jj + i + 1
            if abs(time_series[j] - ts1) < tolerance:  # distance between two vectors
                curr[jj] = prev[jj] + 1
                temp_ts_length = min(sample_length, curr[jj])
                for m in range(int(temp_ts_length)):
                    A[m] += 1
                    if j < n - 1:
                        B[m] += 1
            else:
                curr[jj] = 0
        for j in range(nj):
            prev[j] = curr[j]

    N = n * (n - 1) / 2
    B = np.vstack(([N], B[:sample_length - 1]))
    A_B = np.vstack((A, B))

    return A_B

def refined_composite_multiscale_permutation_entropy(signal, scale, emb_dim, delay):
    """ Calculate refined compositie multiscale permutation entropy.
    Arguments:
        signal: input signal,
        scale: coarse graining scale,
        emd_dim: embedding dimension,
        delay: time delay
    Return:
        rcmpe: refined compositie multiscale permutation entropy value of the signal
    """
    probs = []
    for i in range(scale):
        cg_signal = coarse_graining(signal, i + 1)
        tmp_prob = permutation_frequency(cg_signal, emb_dim, delay)
        probs.append(tmp_prob)
    prob = np.mean(probs, axis=0)
    prob = list(filter(lambda p: p != 0., prob))
    rcmpe = -1 * np.sum(prob * np.log(prob))

    return rcmpe



def permutation_frequency(signal, emb_dim, delay):
    """ Calculate permutation frequency.
    Arguments:
        signal: input signal,
        emb_dim: embedding dimension,
        delay: time delay
    Return:
        prob: permutation frequency of the signal
    """
    length = len(signal)
    permutations = np.array(list(itertools.permutations(range(emb_dim))))
    count = np.zeros(len(permutations))
    for i in range(length - delay * (emb_dim - 1)):
        motif_index = np.argsort(signal[i:i + delay * emb_dim:delay])
        for k, perm in enumerate(permutations):
            if (perm == motif_index).all():
                count[k] += 1

    prob = count / sum(count)

    return prob


# let the m and scale to become paprameters, not use for now 
def RCMSE_combo(signals,max_m,max_scale):
    ''' Preprocessing for EEG signals '''

    # trans_signals = np.transpose(signals)
    # max_scale = 3
    # max_m = 3
    feature_rcmse = np.zeros(( max_m,max_scale))
    for i in range(max_m):
        for j in range(max_scale):
            feature_rcmse[i,j]=RC_composite_multiscale_entropy(signals,i+1,j+1,i+1, None)

    feature_rcmse = feature_mse.flatten()
    features=np.array([])
    features=np.append(features,feature_rcmse)

    return features

'''
Multivariate Multi-Scale Entropy implementation

M the embedding vector
r the time lag vector, always set to 1
'''
def MMSE(old_time_series, M, r, scale, tor=None):
    if tor is None:
        tor = 0.1 * np.std(old_time_series[0])
    time_series = np.array([])
    for i in range(old_time_series.shape[0]):
        tmp = np.array(util_granulate_time_series(old_time_series[i],scale))
        time_series = np.vstack((time_series,tmp)) if time_series.size else tmp

        
    m_count = 0.0
    m_1_count = 0.0
    channel_num = time_series.shape[0]
    vec_1 = np.zeros(sum(M))
    vec_2 = np.zeros(sum(M))

    len_xm = len(time_series[0]) - max(M) * max(r)
    for i in range(len_xm):
        for j in range(i + 1, len_xm):

            for k in range(channel_num - 1):
                vec_1[sum(M[:k]):sum(M[:k + 1])] = time_series[k][i:i + M[k]]
                vec_2[sum(M[:k]):sum(M[:k + 1])] = time_series[k][j:j + M[k]]
            maxnorm = np.max(np.abs(vec_1 - vec_2))
            if maxnorm < tor:
                m_count += 1
                for c in range(channel_num):
                    diff = abs(time_series[c][i + M[c]] - time_series[c][j + M[c]])
                    if diff < tor:
                        m_1_count += 1
    m_count = m_count / (len_xm * (len_xm - 1))

    m_1_count = m_1_count / (len_xm * (len_xm - 1) * channel_num * channel_num)

    return -log((float(m_1_count + 0.00001)) / float(m_count + 0.00001))



