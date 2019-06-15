
from argparse import ArgumentParser
import itertools
import os
import time
import numpy as np



def coarse_graining(signal, scale):
    """Coarse-graining the signal.
    Arguments:
        signal: original signal,
        scale: desired scale
    Return:
        new_signal: coarse-grained signal
    """
    new_length = int(np.fix(len(signal) / scale))
    new_signal = np.zeros(new_length)
    for i in range(new_length):
        new_signal[i] = np.mean(signal[i * scale:(i + 1) * scale])

    return new_signal


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


def multiscale_permutation_entropy(signal, scale, emb_dim, delay):
    """ Calculate multiscale permutation entropy.
    Arguments:
        signal: input signal,
        scale: coarse graining scale,
        emd_dim: embedding dimension,
        delay: time delay
    Return:
        mpe: multiscale permutation entropy value of the signal
    """
    cg_signal = coarse_graining(signal, scale)
    prob = permutation_frequency(cg_signal, emb_dim, delay)
    prob = list(filter(lambda p: p != 0., prob))
    mpe = -1 * np.sum(prob * np.log(prob))

    return mpe

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


def multivariate_multiscale_permutation_entropy(signals, scale, emb_dim, delay):
    """ Calculate multivariate multiscale permutation entropy.
    Arguments:
        signals: input signals,
        scale: coarse graining scale,
        emb_dim: embedding dimension (m),
        delay: time delay
    Return:
        mvpe: multivariate permutation entropy value of the signal
    """
    num_channels = signals.shape[0]
    length = signals.shape[1]

    new_length = int(np.fix(length / scale))
    cg_signals = np.zeros((num_channels, new_length))
    for c in range(num_channels):
        cg_signals[c] = coarse_graining(signals[c], scale)

    permutations = np.array(list(itertools.permutations(range(emb_dim))))
    count = np.zeros((num_channels, len(permutations)))
    for i in range(num_channels):
        for j in range(new_length - delay * (emb_dim - 1)):
            motif_index = np.argsort(cg_signals[i, j:j + delay * emb_dim:delay])
            for k, perm in enumerate(permutations):
                if (perm == motif_index).all():
                    count[i, k] += 1

    count = [el for el in count.flatten() if el != 0]
    prob = np.divide(count, sum(count))
    mmpe = -sum(prob * np.log(prob))
    return mmpe






def main():
    ''' Main function '''
    # Should write some tests
    print("test")

if __name__ == '__main__':

    main()
