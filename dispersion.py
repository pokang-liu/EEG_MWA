#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Multiscale Dispersion Entropy Implementation
"""

from argparse import ArgumentParser
import itertools
import os
import time

from biosppy.signals import ecg
import numpy as np
from scipy.special import comb
from scipy.stats import norm


def coarse_graining(signal, scale):
    """Coarse-graining the signals.
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


def ncdf_mapping(signal):
    """Map the signal into y from 0 to 1 with NCDF.
    Arguments:
        signal: original signal
    Return:
        mapped_signal: mapped signal
    """
    length = len(signal)
    mean = np.mean(signal)
    std = np.std(signal) if np.std(signal) != 0 else 0.001
    ncdf = norm(loc=mean, scale=std)
    mapped_signal = np.zeros(length)
    for i in range(length):
        mapped_signal[i] = ncdf.cdf(signal[i])

    return mapped_signal


def dispersion_frequency(signal, classes, emb_dim, delay):
    """Calculate dispersion frequency.
    Arguments:
        signal: input signal,
        classes: number of classes,
        emb_dim: embedding dimension,
        delay: time delay
    Return:
        prob: dispersion frequency of the signal
    """
    length = len(signal)
    mapped_signal = ncdf_mapping(signal)
    z_signal = np.round(classes * mapped_signal + 0.5)
    dispersions = np.zeros(classes ** emb_dim)
    for i in range(length - (emb_dim - 1) * delay):
        tmp_pattern = z_signal[i:i + emb_dim * delay:delay]
        pattern_index = 0
        for idx, c in enumerate(reversed(tmp_pattern)):
            c = classes if c == (classes + 1) else c
            pattern_index += ((c - 1) * (classes ** idx))

        dispersions[int(pattern_index)] += 1

    prob = dispersions / sum(dispersions)

    return prob


def dispersion_entropy(signal, classes, emb_dim, delay):
    """Calculate dispersion entropy.
    Arguments:
        signal: input signal,
        classes: number of classes,
        emd_dim: embedding dimension,
        delay: time delay
    Return:
        de: dispersion entropy value of the signal
    """
    prob = dispersion_frequency(signal, classes, emb_dim, delay)
    prob = list(filter(lambda p: p != 0., prob))
    de = -1 * np.sum(prob * np.log(prob))

    return de


def multiscale_dispersion_entropy(signal, scale, classes, emb_dim, delay):
    """ Calculate multiscale dispersion entropy.
    Arguments:
        signal: input signal,
        scale: coarse graining scale,
        classes: number of classes,
        emd_dim: embedding dimension,
        delay: time delay
    Return:
        mde: multiscale dispersion entropy value of the signal
    """
    cg_signal = coarse_graining(signal, scale)
    prob = dispersion_frequency(cg_signal, classes, emb_dim, delay)
    prob = list(filter(lambda p: p != 0., prob))
    mde = -1 * np.sum(prob * np.log(prob))

    return mde


def refined_composite_multiscale_dispersion_entropy(signal, scale, classes, emb_dim, delay):
    """ Calculate refined compositie multiscale dispersion entropy.
    Arguments:
        signal: input signal,
        scale: coarse graining scale,
        classes: number of classes,
        emd_dim: embedding dimension,
        delay: time delay
    Return:
        rcmde: refined compositie multiscale dispersion entropy value of the signal
    """
    probs = []
    for i in range(scale):
        cg_signal = coarse_graining(signal, i + 1)
        tmp_prob = dispersion_frequency(cg_signal, classes, emb_dim, delay)
        probs.append(tmp_prob)
    prob = np.mean(probs, axis=0)
    prob = list(filter(lambda p: p != 0., prob))
    rcmde = -1 * np.sum(prob * np.log(prob))

    return rcmde


def multivariate_multiscale_dispersion_entropy(signals, scale, classes, emb_dim, delay):
    """ Calculate multivariate multiscale dispersion entropy.
    Arguments:
        signals: input signals,
        scale: coarse graining scale,
        classes: number of classes,
        emb_dim: embedding dimension,
        delay: time delay
    Return:
        mvmde: multivariate multiscale dispersion entropy value of the signal
    """
    num_channels = signals.shape[0]
    length = signals.shape[1]
    z_signals = np.zeros((num_channels, int(np.fix(length / scale))))
    for i, sc in enumerate(signals):
        cg_signals = coarse_graining(sc, scale)
        mapped_signals = ncdf_mapping(cg_signals)
        z_signals[i] = np.round(classes * mapped_signals + 0.5)

    dispersion = np.zeros(classes ** emb_dim)
    num_patterns = (length - (emb_dim - 1) * delay) * \
        comb(emb_dim * num_channels, emb_dim)
    for i in range(length - (emb_dim - 1) * delay):
        mv_z_signals = z_signals[:, i:i + emb_dim * delay:delay].flatten()
        for tmp_pattern in itertools.combinations(mv_z_signals, emb_dim):
            pattern_index = 0
            for idx, c in enumerate(reversed(tmp_pattern)):
                c = classes if c == (classes + 1) else c
                pattern_index += ((c - 1) * (classes ** idx))

            dispersion[int(pattern_index)] += 1

    prob = dispersion / num_patterns
    prob = list(filter(lambda p: p != 0., prob))
    mvmde = -1 * np.sum(prob * np.log(prob))

    return mvmde
