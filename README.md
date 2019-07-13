Mental workload Assessment Using EEG
===

## Requirement

Python == 3.6.5  
numpy == 1.16.0
scipy == 1.2.0
scikit-learn == 0.20.0

## Introduction
This repository is based on an open-access brain-imaging dataset [1], which consists of 28-channels EEG according to the international 10-5 system. 26 subjects performed Nbacks task experiment. Our goal is to determine MW (0-back v.s. 3-back).


preprocess.py, feature_extraction.py and feature_select_classify.py form a basic machine learning process.

Several EEG features in Time, Frequency, Wavelet, Complexity, and Entropy domain are available in other .py files.



## file description

### preprocess.py
Preprocess the raw EEG data (cnt_nback.mat)and the markers (mnk_nback.mat), generate training 

### feature_extraction.py
Extract time, frequency, wavelet, complexity, entropy domains EEG features, uncomment the features you want to extract in the source code.

The feature and the corresponding name (feature_log) would store in pickle format.




#### feature_select_classify.py

Feature selection and Classification

-    Use the argument to enable RFE feature selection
-    Modify the RFE kernel and classifier in "train" and "plot_classifier_result" functions to apply different clssifiers



### utils.py
The wraper function for feature extraction and other support functions



### Features algorithm packages

#### TimeFeat
-    Mean, std, power, mean of {1st , 2nd } derivative

#### FreqFeat
-    power spectral density, spectral entropy in alpha, beta, gamma, theta band
-    std, mean of {power, absolute value} ratio of absolute mean values (RAM)
#### WSE_HSE
-    Walsh spectral entropy, Haar spectral entropy
#### dispersion 
-    Dispersion Entropy, Refined Composite Multiscale Dispersion Entropy
#### permutation
-    Permutation Entropy, Refined Composite Multiscale Permutation Entropy, Multivariate Multi-Scale Permutation Entropy
#### RCMSE
-    Multi-Scale Entropy, Refined Composite Multi-Scale Entropy, Multivariate Multi-Scale Entropy
#### FD
-    Higuchi Fractal Dimension, Katz Fractal Dimension, Petrosian Fractal Dimension
#### morph
-    Opening Pattern Spectrum, Closing Pattern Spectrum, Curve length, Number of peak, Non-linear energy



## Usage

```bash
python3 preprocess.py --input [EEG_DATA_DIRECTORY] --output [OUTPUT PICKLE FILE OF PROCESSD DATA]


python3 feature_extraction.py --input [EEG_DATA_DIRECTORY] --output [OUTPUT FEATURE PICKLE DIRECTORY]


python3 feature_select_classify.py 


```


## About data format
```bash
mrk.mat label explanation
            16: 0-back target
            48: 2-back target
            64: 2-back non-target
            80: 3-back target
            96: 3-back non-target
            112: 0-back session
            128: 2-back session
            144: 3-back session
```

```bash
channels_log = np.array(['Fp1', 'AFF5h', 'AFz', 'F1', 'FC5', 
                'FC1', 'T7', 'C3', 'Cz', 'CP5', 'CP1', 
                'P7', 'P3', 'Pz', 'POz', 'O1', 'Fp2', 
                'AFF6h', 'F2', 'FC2', 'FC6', 'C4', 'T8', 
                'CP2', 'CP6', 'P4', 'P8', 'O2'])# 'HEOG', 'VEOG']
```



                
```bash
# correct 0 back (value: 2) idx: 2 4 6 10 12 17 18 23 25
# correct 2 back (value: 0) idx: 1 3 8 9 14 16 20 22 24
# correct 3 back (value: 1) idx: 0 5 7 11 13 15 19 21 26
```

## Reference

-    [1] J. Shin, A. von Lühmann, D. Kim, J. Mehnert, H.-J. Hwang, and K.-R.Müller, “Simultaneous acquisition of eeg and nirs during cognitive tasks for an open access dataset,” in Scientific data, 2018.
-    [2] R. Jenke, A. Peer, and M. Buss, “Feature extraction and selection for emotion recognition from eeg,” IEEE Transactions on Affective Computing, vol. 5, no. 3, pp. 327–339, Jul. 2014.
    - A. Subasi, “Eeg signal classification using wavelet feature extraction and a mixture of expert model,” Expert Systems with Applications, vol. 32, no. 4, pp. 1084 – 1093, 2007.
    - [3] R. Jenke, A. Peer, and M. Buss, “Feature extraction and selection for emotion recognition from eeg,” IEEE Transactions on Affective Computing, vol. 5, no. 3, pp. 327–339, Jul. 2014.
-    [4] T. Higuchi, “Approach to an irregular time series on the basis of the fractal theory,” Physica D: Nonlinear Phenomena, vol. 31, pp. 277–283, Jun. 1988.
-    [5] L. Han, L. Zhang, J. Yang, M. Li, and J. Xu, “Method for eeg feature extraction based on morphological pattern spectrum,” in 2009 International Conference on Signal Acquisition and Processing, Apr.2009, pp. 68–72.
-    [6] C. Bandt and B. Pompe, “Permutation entropy: A natural complexity measure for time series,” Phys. Rev. Lett., vol. 88, p. 174102, Apr. 2002.
-    [7] A. Humeau-Heurtier, C.-W. Wu, and S.-D. Wu, “Refined composite multiscale permutation entropy to overcome multiscale permutation entropy length dependence,” IEEE Signal Processing Letters, vol. 22, pp. 1–1, Dec. 2015.
-    [8] F. C. Morabito, D. Labate, F. La Foresta, A. Bramanti, G. Morabito, and I. Palamara, “Multivariate multi-scale permutation entropy for complexity analysis of alzheimer’s disease eeg,” Entropy, vol. 14, no. 7, pp. 1186–1202, 2012.
- [9] J. Onton, A. Delorme, and S. Makeig, “Frontal midline eeg dynamics during working memory,” NeuroImage, vol. 27, pp. 341–56, Sep. 2005
