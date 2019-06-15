from scipy.io import loadmat
from os.path import basename
import numpy as np
import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import RFE
from sklearn.svm import LinearSVC
from scipy.stats import rankdata
from xgboost import XGBClassifier
import argparse
import matplotlib.pyplot as plt


# algorithm packages
import FreqFeature as ff
import TimeFeature as tf
import EntroFeature as ef
import WSE_HSE as wh
import dispersion as dis
import permutation as per
import morph
import RCMSE 
import feature_extract as f
import FD



channels_log = np.array(['Fp1', 'AFF5h', 'AFz', 'F1', 'FC5', 
                'FC1', 'T7', 'C3', 'Cz', 'CP5', 'CP1', 
                'P7', 'P3', 'Pz', 'POz', 'O1', 'Fp2', 
                'AFF6h', 'F2', 'FC2', 'FC6', 'C4', 'T8', 
                'CP2', 'CP6', 'P4', 'P8', 'O2'])# 'HEOG', 'VEOG']

asym_idx1 = [0 ,1 ,3 ,4 ,5 ,6 ,7 ,9 ,10,11,12,15]
asym_idx2 = [16,17,18,20,19,22,21,24,23,26,25,27]

frontal_channels = ['Fp1', 'Fp2', 'AFz', 'AFF5h', 'AFF6h',
                    'F1', 'F2', 'FC5', 'FC6', 'FC1', 'FC2']
back_channels = ['T7', 'C3', 'Cz', 'CP5', 'CP1',
                'P7', 'P3', 'Pz', 'POz', 'O1', 'C4', 'T8', 
                'CP2', 'CP6', 'P4', 'P8', 'O2']
#used_channels = frontal_channels#['Fp1']
#used_channels = back_channels#['Fp1']
#used_channels = ['Fp1', 'Fp2']
used_channels = channels_log
def sub_baseline(x):
    for subject in range(len(x)):
        for trial in range(len(x[subject])):
            for channel in range(len(x[subject][trial])):
                x[subject][trial][channel] = (x[subject][trial][channel] - np.mean(x[subject][trial][channel]))# / np.std(x[subject][trial][channel])
    return x

def store_feature(f, log, file_name):
    dic = {}
    dic['f'] = f
    dic['log'] = log
    pickle_op(file_name, 'w', x = dic)

def fs_rank(feature, y, fs_type):
    if(fs_type == 'fcs'):
        return fcs(feature, y)
    elif(fs_type == 'rfe'):
        return rfe(feature, y)
    else:
        exit()









def psd_feature_extract(x, file_name):
    #    x: 26 subjects x 27 trials x 30 channels x 6080 samples
    theta = [4,7]
    alpha = [8,13]
    beta  = [14,29]
    gamma = [30,45]
    bands = [theta, alpha, beta, gamma]
    bands_log = ['theta', 'alpha', 'beta', 'gamma']

    channels_log = ['Fp1', 'AFF5h', 'AFz', 'F1', 'FC5', 
                    'FC1', 'T7', 'C3', 'Cz', 'CP5', 'CP1', 
                    'P7', 'P3', 'Pz', 'POz', 'O1', 'Fp2', 
                    'AFF6h', 'F2', 'FC2', 'FC6', 'C4', 'T8', 
                    'CP2', 'CP6', 'P4', 'P8', 'O2', 'HEOG', 'VEOG']

    asym_idx1 = [0 ,1 ,3 ,4 ,5 ,6 ,7 ,9 ,10,11,12,15]
    asym_idx2 = [16,17,18,20,19,22,21,24,23,26,25,27]

    x_psd = [] # 32 x 40 x 128 x (1 or 29)
    psd_log = []

    for subject in range(len(x)):
        feature_subject = []
        for trial in range(len(x[subject])):
            feature_trial = []
            for channel in range(len(x[subject][trial])-2): # no HEOG, VEOG
                feature_channel = []
                for i, band in enumerate(bands):
                    feature_trial.append(ff.psd(x[subject][trial][channel],
                                         window = int(len(x[subject][trial][channel]) / 200),
                                         overlap = 0,
                                         start_freq = band[0],
                                         end_freq = band[1],
                                         Fs = 200))
                    if((subject == 0) & (trial == 0)):
                        psd_log.append('{}_{}_{}'.format(channels_log[channel], bands_log[i], 'psd'))
            feature_subject.append(np.array(feature_trial))
        x_psd.append(np.array(feature_subject))

    print(psd_log)
    psd_log = np.array(psd_log)
    x_psd = np.array(x_psd)
    x_psd = x_psd.reshape(x_psd.shape[0], x_psd.shape[1], -1)

    store_feature(x_psd, psd_log, file_name)
    return x_psd, psd_log

def feature_extract(x, func, func_name, file_name, type, scale = 3, match = 3):
    '''
        x: 26 subjects x 27 trials x 30 channels x 6080 samples
    '''
    # used for freq domain feature
    #theta = [4,7]
    #alpha = [8,13]
    #beta  = [14,29]
    #gamma = [30,45]
    theta = [7,10]
    alpha = [11,19]
    beta  = [20,29]
    gamma = [30,45]
    bands = [theta, alpha, beta, gamma]
    bands_log = ['theta', 'alpha', 'beta', 'gamma']

    log = []
    feature = []
    for subject in range(len(x)): #26
        print('subject ',subject)
        feature_subject = [] 
        # print('len(x[subject]) ',len(x[subject]))
        for trial in range(len(x[subject])): # 27
            print('trial ',trial)
            feature_trial = []
            for channel in range(len(x[subject][trial])-2): #30
                
                if(type == 'time'):
                    if(func_name == 'T_hoc'):
                        for iii in range(10):
                            feature_trial.append(func(x[subject][trial][channel], order = iii))
                    elif(func_name[0] == 'W'):
                            wavelet_feature = func(x[subject][trial][channel])
                            for k in range(len(wavelet_feature)):
                                feature_trial.append(wavelet_feature[k])

                    else:
                        feature_trial.append(func(x[subject][trial][channel]))

                    if((subject == 0) & (trial == 0)):
                        if(func_name == 'T_hoc'):
                            for iii in range(10):
                                log.append('{}_{}_{}'.format(channels_log[channel], func_name, iii))
                        elif(func_name == 'W_RAM'):
                            log.append('{}_{}_{}'.format(channels_log[channel], func_name,'cA4/cD4'))
                            log.append('{}_{}_{}'.format(channels_log[channel], func_name,'cD4/cD3'))
                            log.append('{}_{}_{}'.format(channels_log[channel], func_name,'cA3/cD2'))
                        elif(func_name[0] == 'W'):
                            log.append('{}_{}_{}'.format(channels_log[channel], func_name,'cA4'))
                            log.append('{}_{}_{}'.format(channels_log[channel], func_name,'cD4'))
                            log.append('{}_{}_{}'.format(channels_log[channel], func_name,'cD3'))        
                            log.append('{}_{}_{}'.format(channels_log[channel], func_name,'cD2'))        
                        else:
                            log.append('{}_{}'.format(channels_log[channel], func_name))
                elif(type == 'freq'):
                    #sig = []
                    #for i in range(len(x[subject][trial][channel])):
                    #    #print(type(float(x[subject][trial][channel][i])))
                    #    sig.append(float(x[subject][trial][channel][i]))
                    #sig = np.array(sig)
                    #print(type(sig))
                    sig = x[subject][trial][channel]
                    for i, band in enumerate(bands):
                        feature_trial.append(func(sig, 
                                                  start_freq = band[0],
                                                  end_freq = band[1], 
                                                  Fs = 200))
                        if((subject == 0) & (trial == 0)):
                            log.append('{}_{}_{}'.format(channels_log[channel], func_name, bands_log[i]))
                elif(type == 'en'):
                    print(subject, trial, channel)
                    mse = ef.multiscale_entropy(x[subject][trial][channel], sample_length = 3, maxscale = 5)
                    feature_trial.append(mse[0])
                    feature_trial.append(mse[1])
                    feature_trial.append(mse[2])
                    feature_trial.append(mse[3])
                    feature_trial.append(mse[4])
                    if((subject == 0) & (trial == 0)):
                        log.append('{}_{}_0'.format(channels_log[channel], func_name)) 
                        log.append('{}_{}_1'.format(channels_log[channel], func_name)) 
                        log.append('{}_{}_2'.format(channels_log[channel], func_name)) 
                        log.append('{}_{}_3'.format(channels_log[channel], func_name)) 
                        log.append('{}_{}_4'.format(channels_log[channel], func_name)) 
                        ####################
                elif(type == 'new'):

                    if(func_name == 'RCMSE'):
                        m = np.array([2,3,4])

                        scale = [11,13,15,17,19]

                        ## add new entropy domian features!!
                        for iii, ii in enumerate(m):
                            print('m =',ii)
                            for jjj, jj in enumerate(scale):
                                print('scale=',jj)
                               
                                rcmse = RCMSE.RC_composite_multiscale_entropy(x[subject][trial][channel],ii+1,jj+1,ii)
                                print('rcmse:',rcmse)
                                feature_trial.append(rcmse)
                                if((subject == 0) & (trial == 0)):
                                    log.append('{}_{}_m{}_s{}'.format(channels_log[channel], func_name, ii, jj))
                       
                    elif(func_name == 'RCMPE'):
                    # #################  to do:
                        emb_dim = [3,4,5]
                        scale = [ 21,23,25,27,29]
                        for dd, d in enumerate(emb_dim):
                            print('d =',d)
                            for ss, s in enumerate(scale):
                                print('s =',s)
                                rcmpe = per.refined_composite_multiscale_permutation_entropy(x[subject][trial][channel],s,d,1)
                                print('rcmpe:',rcmpe)

                                feature_trial.append(rcmpe)
                                if((subject == 0) & (trial == 0)):
                                    log.append('{}_{}_d{}_s{}'.format(channels_log[channel], func_name, d, s))   


                    elif(func_name == 'MDE'): 
                        emb_dim = [3,4,5]
                        scale = [20,25,30]
                        # classes = [3,6,9]
                        for dd, d in enumerate(emb_dim):
                            print('d =',d)
                            for ss, s in enumerate(scale):
                                print('s =',s)
                                mde = dis.refined_composite_multiscale_dispersion_entropy(x[subject][trial][channel],s,4,d,1)
                                print('mde:',mde)
                                feature_trial.append(mde)
                                if((subject == 0) & (trial == 0)):
                                    log.append('{}_{}_d{}_s{}'.format(channels_log[channel], func_name, d, s))   
                    elif(func_name == "WSE"):
                        wse = func(x[subject][trial][channel])
                        print('wse',wse)
                        feature_trial.append(wse)
                        if((subject == 0) & (trial == 0)):

                            log.append('{}_{}'.format(channels_log[channel], func_name))   
                   
                    elif(func_name == "HSE"):
                        hse = func(x[subject][trial][channel])
                        print('hse',hse)
                       
                        feature_trial.append(hse)
                        if((subject == 0) & (trial == 0)):
    
                            log.append('{}_{}'.format(channels_log[channel], func_name))   


                    elif(func_name == "KFD"):
                        kfd = func(x[subject][trial][channel])
                        print('kfd',kfd)
                       
                        feature_trial.append(kfd)
                        if((subject == 0) & (trial == 0)):
       
                            log.append('{}_{}'.format(channels_log[channel], func_name))   


                    elif(func_name == "PFD"):
                        pfd = func(x[subject][trial][channel])
                        print('pfd',pfd)
                       
                        feature_trial.append(pfd)
                        if((subject == 0) & (trial == 0)):
                       
                            log.append('{}_{}'.format(channels_log[channel], func_name))   


                    elif(func_name == "HFD"):
                        hfd = func(x[subject][trial][channel])
                        print('hfd',hfd)
                       
                        feature_trial.append(hfd)
                        if((subject == 0) & (trial == 0)):
                       
                            log.append('{}_{}'.format(channels_log[channel], func_name))   

                    elif(func_name == 'CLOSE'):
                    # #################  to do:
                        se = [2,3,4]
                        scale = [ 1,2,3,4,5]
                        for sese, se in enumerate(se):
                            print('se =',se)
                            for ss, s in enumerate(scale):
                                print('s =',s)
                                close = morph.close_pattern_spectrum(x[subject][trial][channel],se,s)
                                print('close:',close)

                                feature_trial.append(close)
                                if((subject == 0) & (trial == 0)):
                                    log.append('{}_se{}_s{}'.format(channels_log[channel],se,s))


                    elif(func_name == 'OPEN'):
                    # #################  to do:
                        se = [2,3,4]
                        scale = [ 1,2,3,4,5]
                        for sese, se in enumerate(se):
                            print('se =',se)
                            for ss, s in enumerate(scale):
                                print('s =',s)
                                open_ = morph.open_pattern_spectrum(x[subject][trial][channel],se,s)
                                print('open:',open_)

                                feature_trial.append(open_)
                                if((subject == 0) & (trial == 0)):
                                    log.append('{}_se{}_s{}'.format(channels_log[channel],se,s ))   

                    elif(func_name == "CURVE"):
                        curve = func(x[subject][trial][channel])
                        print('curve',curve)
                       
                        feature_trial.append(curve)
                        if((subject == 0) & (trial == 0)):
                       
                            log.append('{}_{}'.format(channels_log[channel], func_name))   

 
                    elif(func_name == "PEAK"):
                        peak = func(x[subject][trial][channel])
                        print('peak',peak)
                       
                        feature_trial.append(peak)
                        if((subject == 0) & (trial == 0)):
                       
                            log.append('{}_{}'.format(channels_log[channel], func_name))   

                    elif(func_name == "NONLINEAR_ENG"):
                        eng = func(x[subject][trial][channel])
                        print('eng',eng)
                       
                        feature_trial.append(eng)
                        if((subject == 0) & (trial == 0)):
                       
                            log.append('{}_{}'.format(channels_log[channel], func_name))   



            feature_subject.append(np.array(feature_trial))
        feature.append(np.array(feature_subject))
    feature, log = np.array(feature), np.array(log)

    store_feature(feature, log, file_name)
    return feature, log
def normalize(x): # normalize over per subject
    # x shape = 26 x 27 x 120
    x = x.reshape(x.shape[0], x.shape[1], -1)
    print('shape1',x.shape)
    x_norm = []
    for subject in range(len(x)):
        x_subject_t = np.transpose(x[subject]) # 120 x 27
        for i in range(len(x_subject_t)): # x_subject_t[i].shape = (27,)
            x_subject_t[i] = (x_subject_t[i] - np.mean(x_subject_t[i])) / np.std(x_subject_t[i])
        x_norm.append(np.transpose(x_subject_t))
        #print('x_norm shape',len(x_norm))

    return np.array(x_norm)

def pickle_op(file_name, mode, x = None):
    if(mode == 'w'):
        with open('{}'.format(file_name),'wb') as f:
            pickle.dump(x, f)
        return None
    elif(mode == 'r'):
        with open('{}'.format(file_name),'rb') as f:
            output = pickle.load(f)
        return output
    else:
        print('Wrong mode type!!!!') 

def take_0_3_back(x, y):
    
    # correct 0 back (value: 2) idx: 2 4 6 10 12 17 18 23 25
    # correct 2 back (value: 0) idx: 1 3 8 9 14 16 20 22 24
    # correct 3 back (value: 1) idx: 0 5 7 11 13 15 19 21 26

    # y shape: 26 x 27
    # x shape: 26 x 27 x 120
    # return shape = x: 26 x 18 x 120, y: 26 x 18
    x_sub = []
    y_sub = []
    two_back_idx =  [1,3,8,9,14,16,20,22,24] 
    for subject in range(len(x)):
        #if(args.input_type == 'whole'):
        #    y_sub.append(np.delete(y[subject], range(9,18), axis = 0))
        #    x_sub.append(np.delete(x[subject], range(9,18), axis = 0))
        #elif(args.input_type == 'seg'):
        #y_sub.append(np.delete(y[subject], range(int(len(y[subject])/3*1),int(len(y[subject])/3*2)), axis = 0))
        #x_sub.append(np.delete(x[subject], range(int(len(x[subject])/3*1),int(len(x[subject])/3*2)), axis = 0))
        
        
        y_sub.append(np.delete(y[subject],two_back_idx , axis = 0))
        x_sub.append(np.delete(x[subject],two_back_idx , axis = 0))
    
    return np.array(x_sub), np.array(y_sub)

def take_0_2_back(x, y):

   
    # y shape: 26 x 27
    # x shape: 26 x 27 x 120
    # return shape = x: 26 x 18 x 120, y: 26 x 18
    x_sub = []
    y_sub = []
    three_back_idx = [0, 5, 7, 11, 13, 15, 19, 21, 26] 

    for subject in range(len(x)):
        #if(args.input_type == 'whole'):
        #    y_sub.append(np.delete(y[subject], range(9,18), axis = 0))
        #    x_sub.append(np.delete(x[subject], range(9,18), axis = 0))
        #elif(args.input_type == 'seg'):
        #y_sub.append(np.delete(y[subject], range(int(len(y[subject])/3*1),int(len(y[subject])/3*2)), axis = 0))
        #x_sub.append(np.delete(x[subject], range(int(len(x[subject])/3*1),int(len(x[subject])/3*2)), axis = 0))


        y_sub.append(np.delete(y[subject],three_back_idx , axis = 0))
        x_sub.append(np.delete(x[subject],three_back_idx , axis = 0))

    return np.array(x_sub), np.array(y_sub)

def take_2_3_back(x, y):



    # y shape: 26 x 27
    # x shape: 26 x 27 x 120
    # return shape = x: 26 x 18 x 120, y: 26 x 18
    x_sub = []
    y_sub = []
    zero_back_idx = [2, 3, 8, 9, 14, 16, 20, 22, 24]

    for subject in range(len(x)):
        #if(args.input_type == 'whole'):
        #    y_sub.append(np.delete(y[subject], range(9,18), axis = 0))
        #    x_sub.append(np.delete(x[subject], range(9,18), axis = 0))
        #elif(args.input_type == 'seg'):
        #y_sub.append(np.delete(y[subject], range(int(len(y[subject])/3*1),int(len(y[subject])/3*2)), axis = 0))
        #x_sub.append(np.delete(x[subject], range(int(len(x[subject])/3*1),int(len(x[subject])/3*2)), axis = 0))
        y_sub.append(np.delete(y[subject],zero_back_idx , axis = 0))
        x_sub.append(np.delete(x[subject],zero_back_idx , axis = 0))       
    return np.array(x_sub), np.array(y_sub)



def get_used_feature_idx(feature_log, used_channels):
    idx = []
    for i in range(len(used_channels)):
        for j in range(len(feature_log)):
            if(used_channels[i] in feature_log[j]):
                idx.append(j)
    return np.array(idx)




def channel_select(feature, log, ch_score, num_channel):
    idx_sort = rankdata(-np.array(ch_score), method = 'min')
    wanted_channel_idx = (idx_sort < num_channel+1)
    wanted_channel = channels_log[wanted_channel_idx]
    rem_idx = []
    for i in range(len(log)):
        ch = log[i].split('_')[0]
        if(ch not in wanted_channel):
            rem_idx.append(i)
    sel_feature = np.delete(feature, rem_idx, axis = 2)
    sel_log = np.delete(log, rem_idx)

    return sel_feature, sel_log

def feature_select(feature, log, f_rank, num_feature):
    sel_feature = f_rank < num_feature 
    print(feature.shape)
    print(len(sel_feature))
    return feature[:,:,sel_feature], log[sel_feature]

def channel_importance(rank, log, fs_type, number = None):
    def _get_previous(score, number = None):
        if number == None:
            number = len(score) - 1
        thres = np.sort(score)[-number]
        for i in range(len(score)):
            if(score[i] < thres):
                score[i] = 0
        return np.array(score)

    small_imp_rank = rank  # len 112
    rank = len(log) - small_imp_rank
    ch_score = [0] * 28 # len 28
    ch_log = np.array(channels_log)
    cm = plt.cm.get_cmap('RdYlBu')

    feature = log[0].split('_')[1] + '_' + log[0].split('_')[2]
    for i in range(len(rank)):
        ch = log[i].split('_')[0]
        idx = np.where(ch_log == ch)[0][0]
        ch_score[idx] = ch_score[idx] + rank[i]

    ch_rank = rankdata(-np.array(ch_score), method = 'min')
    ch_score = _get_previous(ch_score, number)

    pos = pickle_op('saved_pickle/channel_pos.p', 'r')
    for i in range(len(ch_score)):
        sc = plt.scatter(pos[i][0], pos[i][1], c = ch_score[i], cmap = cm, vmin=np.min(ch_score), vmax=np.max(ch_score), s = 500)
        plt.text(pos[i][0]-0.15, pos[i][1], '{}_{}_{}'.format(ch_log[i], ch_score[i], ch_rank[i]), fontsize=7)
    plt.colorbar(sc)
    plt.title('{}'.format(feature))
    plt.savefig('fig/ch_importance/{}/{}.jpg'.format(fs_type, feature))
    #plt.show()
    plt.clf()
    return ch_score




from scipy.signal import butter, lfilter, find_peaks

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def bandpass(x):    
    fs = 200.0
    lowcut = 4.0
    highcut = 45.0

    x_f = []
    for channel in range(len(x)):
        x_f.append(butter_bandpass_filter(x[channel], lowcut, highcut, fs, order= 2))
    return np.array(x_f)

def remove_peak(x, idx):
    new_x = []
    for i in range(len(x)):
        if(i in idx):
            a = x[i-5:i+5]
            m_idx = np.random.randint(0,5)
            v = np.sort(a)[m_idx]
            new_x.append(v)
        else:
            new_x.append(x[i])

    return np.array(new_x)
def plot_freq_binRMS(x, y, channel):
    """
        x shape = 26 x 27 x 30 x 6080
    """
    #x, y = take_0_3_back(x,y)
    
    #subject = 10
    #channel = 0
    band = np.arange(5,40,0.3)
    idx = []
    for i in range(len(band)):
        if(((band[i]) > 10.0) & ((band[i])<11.0)):
            idx.append(i)
    for i in range(len(band)):
        if(((band[i]) > 20.0) & ((band[i])<21.5)):
            idx.append(i)
    for i in range(len(band)):
        if(((band[i]) > 30.7) & ((band[i])<31.7)):
            idx.append(i)
    print(channel)
    rms_list_0 = []
    rms_list_2 = []
    rms_list_3 = []
    for subject in range(len(x)):
        power_list_0 = []
        power_list_2 = []
        power_list_3 = []
        x_EEG = x[subject,:,channel,:] # 27 x 6080


        x_EEG_0 = x_EEG[y[subject] == 0] # 9 x 6080
        x_EEG_2 = x_EEG[y[subject] == 2] # 9 x 6080
        x_EEG_3 = x_EEG[y[subject] == 1] # 9 x 6080
        for i in range(len(x_EEG_0)): # 9 task
            power_list_0.append((ff.bin_power(X = x_EEG_0[i], Band = band, Fs = 200)[1]))
            power_list_2.append((ff.bin_power(X = x_EEG_2[i], Band = band, Fs = 200)[1]))
            power_list_3.append((ff.bin_power(X = x_EEG_3[i], Band = band, Fs = 200)[1]))
        rms_power_0 = np.mean(np.array(power_list_0), axis = 0)
        rms_power_2 = np.mean(np.array(power_list_2), axis = 0)
        rms_power_3 = np.mean(np.array(power_list_3), axis = 0)

        rms_list_0.append(rms_power_0 )
        rms_list_2.append(rms_power_2 )
        rms_list_3.append(rms_power_3 )

    mean_rms_list_0 = remove_peak(np.mean(np.array(rms_list_0), axis = 0),idx)
    mean_rms_list_2 = remove_peak(np.mean(np.array(rms_list_2), axis = 0),idx)
    mean_rms_list_3 = remove_peak(np.mean(np.array(rms_list_3), axis = 0),idx)


    lowf_high = []
    highf_high = []
    for i in range(len(mean_rms_list_0)):
        if(mean_rms_list_0[i] > mean_rms_list_3[i]):
            lowf_high.append(band[i])
        else:
            highf_high.append(band[i])
    plt.plot(band[0:-1], mean_rms_list_0, label = '0-back', color = 'r')
    #plt.plot(band[0:-1], r_mean_rms_list_0, label = '0-back')
    #plt.plot(band[0:-1], mean_rms_list_2, label = '2-back')
    plt.plot(band[0:-1], mean_rms_list_3, label = '3-back', color = 'b')

    plt.scatter(lowf_high, [0.000]*len(lowf_high), color = 'r',s=15)
    plt.scatter(highf_high, [0.000]*len(highf_high), color = 'b',s=15)
    plt.ylim(ymax = 0.0180)
    #plt.plot(band[0:-1], np.mean(rms_list_1, axis = 0), label = '3-back')
    
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Normalized power')
    plt.legend(loc=1, prop={'size': 18})
    #plt.title('{}'.format(channels_log[channel]))
    #plt.show()
    plt.savefig('paper/bin_rms/eps/{}_{}.eps'.format(channel, channels_log[channel]), format ='eps',bbox_inches='tight')
    plt.clf()
    plt.cla()
    plt.close()
    return rms_power_0, rms_power_3

def print_ranking_spacing(rank, feature_log):
    print(np.arange(2,12))
    offset = 0
    mini = 2
    maxi = 12
    for i in range(10):
        print('---ranking between {} and {}---'.format(mini+offset - 1, maxi+offset - 2))
        sel_1 = (rank >= (mini + offset-1))
        sel_2 = (rank <= (maxi + offset-2))
        offset += 10
        sel = (sel_1 & sel_2)
        select_feature = feature_log[sel]
        theta, alpha, beta, gamma = 0, 0, 0, 0
        for j in range(len(select_feature)):
            band = select_feature[j].split('_')[1]
            if(band == 'theta'):
                theta += 1
            if(band == 'alpha'):
                alpha += 1
            if(band == 'beta'):
                beta += 1
            if(band == 'gamma'):
                gamma += 1
        print('Number of theta band: {}'.format(theta))
        print('Number of alpha band: {}'.format(alpha))
        print('Number of beta band: {}'.format(beta))
        print('Number of gamma band: {}'.format(gamma))
def print_location_ranking_spacing(rank, feature_log):
    print(np.arange(2,12))
    offset = 0
    mini = 2
    maxi = 12


    front = ['Fp1', 'Fp2', 'AFz', 'AFF5h', 'AFF6h', 'F1', 'F2', 'FC5', 'FC1', 'FC2', 'FC6'] 
    med = ['T7', 'C3', 'Cz', 'C4', 'T8', 'CP5', 'CP1', 'CP2', 'CP6']
    back = ['P7', 'P3', 'Pz', 'P4', 'P8', 'POz', 'O1', 'O2']
    for i in range(10):
        print('---location ranking between {} and {}---'.format(mini+offset - 1, maxi+offset - 2))
        print(mini+offset-1, maxi+offset-2)
        #print(rank)
        sel_1 = (rank >= (mini + offset-1))
        sel_2 = (rank <= (maxi + offset-2))
        offset += 10
        sel = (sel_1 & sel_2)
        select_feature = feature_log[sel]
        front_num, med_num, back_num = 0, 0, 0
        for j in range(len(select_feature)):
            loc = select_feature[j].split('_')[0]

            if(loc in front):
                front_num += 1
            elif(loc in med):
                med_num += 1
            elif(loc in back):
                back_num += 1
            else:
                print(loc)
        print('Number of front: {}'.format(front_num))
        print('Number of med: {}'.format(med_num))
        print('Number of back: {}'.format(back_num))


def read_feature(file_name):
    a = pickle_op(file_name, 'r')['f']
    b = pickle_op(file_name, 'r')['log']
    if(len(a[0][0])!=len(b)):
        print('file_name ',file_name) 

        print('len(a[0][0]) ',len(a[0][0])) 

    #print('a', a)
        print('len(b) ',len(b)) 
    #print('b ',b)

    # return pickle_op(file_name, 'r')['f'], pickle_op(file_name, 'r')['log']
    return a, b


def rank_feature(x, y,estimatorr):
    #estimator = GaussianNB()
    #XGBClassifier(max_depth = 3)
    #estimator = LinearSVC()
    estimator = estimatorr
    selector = RFE(estimator, 1, step = 1, verbose = 0)
    selector = selector.fit(x, y)

    feature = selector.support_ # return true or false
    feature_ranking = selector.ranking_
    return feature_ranking


def extract_differential(f, logs, func_name, file_name, type = 'minus'):

    asym_log1 = np.array(['Fp1', 'AFF5h', 'F1', 'FC5', 'FC1', 'T7', 'C3', 'CP5', 'CP1', 'P7', 'P3', 'O1'])
    asym_log2 = np.array(['Fp2', 'AFF6h', 'F2', 'FC6', 'FC2', 'T8', 'C4', 'CP6', 'CP2', 'P8', 'P4', 'O2'])
    channels_log = np.array(['Fp1', 'AFF5h', 'AFz', 'F1', 'FC5', 
                'FC1', 'T7', 'C3', 'Cz', 'CP5', 'CP1', 
                'P7', 'P3', 'Pz', 'POz', 'O1', 'Fp2', 
                'AFF6h', 'F2', 'FC2', 'FC6', 'C4', 'T8', 
                'CP2', 'CP6', 'P4', 'P8', 'O2'])# 'HEOG', 'VEOG']

    feature_idx1 = []
    feature_idx2 = []
    log = []
    for i, log1 in enumerate(logs):
        channel1, remain = log1.split('_', 1)[0], log1.split('_', 1)[1]
        if channel1 in asym_log1:
            find_asym_log_idx = np.where(asym_log1 == channel1)[0][0]
            channel2 = asym_log2[find_asym_log_idx]
            log2 = '{}_{}'.format(channel2, remain)

            feature_idx1.append(i)
            feature_idx2.append(np.where(logs == log2)[0][0])
            log.append('{}_{}_{}'.format(channel1, func_name, remain.split('_',1)[1]))


    if(type == 'minus'):
        feature = f[:, :, feature_idx1] - f[:, :, feature_idx2]
    elif(type == 'divide'):
        feature = f[:, :, feature_idx1] / f[:, :, feature_idx2]
    else:
        for i in range(1000000):
            print('error type!!!!!!!!!!!!')
    
    feature, log = np.array(feature), np.array(log)
    store_feature(feature, log, file_name)

    return feature, log

def band_selection(f, logs, band):
    idx_list = []
    band_type = ['alpha', 'beta', 'gamma', 'theta']

    assert band in band_type
    for i in range(len(logs)):
        split = logs[i].split('_')
        if(band in split):
            idx_list.append(i)


    return f[:, :, idx_list], logs[idx_list]


def fcs(x, y):
    # x: 32 x 40 x 128
    # y: 32 x 40
    x = x.copy().reshape(x.shape[0]*x.shape[1], -1)
    y = np.array(y).copy().reshape(-1)

    x_1 = np.transpose(x[y == 1]) #128 x 633
    x_0 = np.transpose(x[y == 0]) #128 x 647

    mean_1 = np.mean(x_1, axis = 1)# 128
    mean_0 = np.mean(x_0, axis = 1)
    std_1 = np.std(x_1, axis = 1)
    std_0 = np.std(x_0, axis = 1)
    
    denominator = np.square(std_1) + np.square(std_0)# down, (128,)
    denominator[denominator == 0] = 0.0001

    nominator = np.square(mean_1 - mean_0).astype(float)
    score = nominator / denominator

    order = np.argsort(-score) # descending order, most important feature indexed 0
    return order
def rfe(x,y):
    x = x.copy().reshape(x.shape[0]*x.shape[1], -1)
    y = np.array(y).copy().reshape(-1)


    estimator = LinearSVC()
    selector = RFE(estimator, 1, step = 1, verbose = 0)
    selector = selector.fit(x, y)

    feature = selector.support_ # return true or false
    feature_ranking = selector.ranking_

    return feature_ranking


def specify_channel(feature, feature_log, wanted_channel):
    wanted_idx = []
    for i in range(len(feature_log)):
        ch = feature_log[i].split('_')[0]
        if(ch in wanted_channel):
            wanted_idx.append(i)
    return feature[:,:,wanted_idx], feature_log[wanted_idx]
### read matlab file
def read_cnt(path):
    '''
        clab: channel label
        fs: sampling rate (200)
        title: 'nback'
        x: signal len (3 tasks) * 30 (# channels)  (388844, 30)
        T: # samples per task
        yUnit: uV
    '''
    file_name = basename(path)
    file_name = file_name[0:len(file_name)-4]
    data = loadmat(path)[file_name]
    clab = data[0, 0]['clab'][0]#.astype('str')
    clab = [clab[i][0] for i in range(len(clab))]
    fs = data[0, 0]['fs'][0][0]
    title = data[0, 0]['title'][0]
    x = data[0, 0]['x']
    T = data[0, 0]['T'][0][0]
    yUnit = data[0, 0]['yUnit'][0]
    #return clab, fs, title, x, T, yUnit
    return x

def read_mnt(path):
    file_name = basename(path)
    file_name = file_name[0:len(file_name)-4]
    data = loadmat(path)[file_name]

    x = data[0, 0]['x'].reshape(-1)
    y = data[0, 0]['y'].reshape(-1)
    pos_3d = data[0, 0]['pos_3d']
    clab = data[0, 0]['clab'][0]#.astype('str')
    clab = [clab[i][0] for i in range(len(clab))]
    box = data[0, 0]['box']
    box_sz = data[0, 0]['box_sz']
    scale_box = data[0, 0]['scale_box']
    scale_box_sz = data[0, 0]['scale_box_sz']

    return x, y, pos_3d, clab, box, box_sz, scale_box, scale_box_sz
def read_mrk(path):
    '''
        time: defines the time points of event in msec
        y: class labels (one hot)
        className: ['0-back target', '2-back target', 
                    '2-back non-target', '3-back target', 
                    '3-back non-target', '0-back session', 
                    '2-back session', '3-back session']
        event: len = 567
            16: 0-back target
            48: 2-back target
            64: 2-back non-target
            80: 3-back target
            96: 3-back non-target
            112: 0-back session
            128: 2-back session
            144: 3-back session
    '''
    file_name = basename(path)
    file_name = file_name[0:len(file_name)-4]
    data = loadmat(path)[file_name]

    time = data[0, 0]['time'][0]
    y = data[0, 0]['y']
    className = data[0, 0]['className'][0]
    className = [className[i][0] for i in range(len(className))]
    event = data[0, 0]['event'][0][0][0].reshape(-1)
    
    #return time, y, className, event
    print('time ', time)
    print('event ', event)
    print("y: ",y)
    return time, event, y


