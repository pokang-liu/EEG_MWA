
import argparse
import numpy as np
import os.path
import time
import pickle

### other files in this repo
import RCMSE 
import dispersion as dis
import permutation as per
import morph
import FD
import FreqFeature as ff
import TimeFeature as tf
import EntroFeature as ef
import utils as f
import WSE_HSE as wh

channels_log = np.array(['Fp1', 'AFF5h', 'AFz', 'F1', 'FC5', 
                'FC1', 'T7', 'C3', 'Cz', 'CP5', 'CP1', 
                'P7', 'P3', 'Pz', 'POz', 'O1', 'Fp2', 
                'AFF6h', 'F2', 'FC2', 'FC6', 'C4', 'T8', 
                'CP2', 'CP6', 'P4', 'P8', 'O2'])# 'HEOG', 'VEOG']

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Extract CNN feature')
    parser.add_argument('-t', '--input_type', help = 'input file type whole or seg', default = 'whole', choices = ['whole', 'seg'])
    parser.add_argument('-input', '--input', help = 'input file', default = 'whole_trial_26_eog')
    parser.add_argument('-output', '--output', help = 'output feature directory', default = '26eog')

    args = parser.parse_args()
    
    dic = f.pickle_op(file_name = 'saved_pickle/{}.p'.format(args.input), mode = 'r')
    y = dic['y'] # 26 x 27
    x = dic['x'] # 26 x 27 x 30 x 6080 or 26 x 513 x 30 x 419
    x = f.sub_baseline(x)
    

    '''
    #-------------------feature extract & store-----------------------------
    #-------------------uncomment the feature you want -----------------------------

    #-------------------time domain----------------------------

    f.feature_extract(x, tf.pfd, 'T_pfd', '{}_feature_pickle/{}/T_pfd.p'.format(args.input_type, args.dir), type = 'time')

    f.feature_extract(x, tf.mean_power, 'T_mean_power', '{}_feature_pickle/{}/T_mean_power.p'.format(args.input_type, args.dir), type = 'time')
    f.feature_extract(x, tf.mean, 'T_mean', '{}_feature_pickle/{}/T_mean.p'.format(args.input_type, args.dir), type = 'time')
    f.feature_extract(x, tf.std, 'T_std', '{}_feature_pickle/{}/T_std.p'.format(args.input_type, args.dir), type = 'time')
    f.feature_extract(x, tf.first_diff, 'T_first_diff', '{}_feature_pickle/{}/T_first_diff.p'.format(args.input_type, args.dir), type = 'time')
    f.feature_extract(x, tf.second_diff, 'T_second_diff', '{}_feature_pickle/{}/T_second_diff.p'.format(args.input_type, args.dir), type = 'time')
     
    #-------------------frequency domain----------------------------

    f.feature_extract(x, ff.psd, 'FT_psd', '{}_feature_pickle/{}/FT_psd.p'.format(args.input_type, args.dir), type = 'freq')
    f.feature_extract(x, ff.stft_power, 'STFT_power', '{}_feature_pickle/{}/STFT_power.p'.format(args.input_type, args.dir), type = 'freq')
    f.feature_extract(x, ff.spectral_entropy, 'FT_se', '{}_feature_pickle/{}/FT_se.p'.format(args.input_type, args.dir), type = 'freq')
    f.feature_extract(x, ef.multiscale_entropy, 'EN_mse', '{}_feature_pickle/EN_mse.p'.format(args.input_type), type = 'en')
    
    ################################
    #-------------------wavelet domain----------------------------

    f.feature_extract(x, ff.waveletF, 'F_wavelet', '{}_feature_pickle/{}/F_wavelet.p'.format(args.input_type, args.dir), type = 'time')
    f.feature_extract(x, ff.psdMAV, 'F_psdMAV', '{}_feature_pickle/{}/F_psdMAV.p'.format(args.input_type, args.dir), type = 'time')

    ##############################
    
    f.feature_extract(x, ff.waveletT_MAV, 'W_MAV', '{}_feature_pickle/{}/W_MAV.p'.format(args.input_type, args.dir), type = 'time')
    f.feature_extract(x, ff.waveletT_AP, 'W_AP', '{}_feature_pickle/{}/W_AP.p'.format(args.input_type, args.dir), type = 'time')
    f.feature_extract(x, ff.waveletT_SD, 'W_SD', '{}_feature_pickle/{}/W_SD.p'.format(args.input_type, args.dir), type = 'time')
    f.feature_extract(x, ff.waveletT_first_diff, 'W_first_diff', '{}_feature_pickle/{}/W_first_diff.p'.format(args.input_type, args.dir), type = 'time')
    f.feature_extract(x, ff.waveletT_second_diff, 'W_second_diff', '{}_feature_pickle/{}/W_second_diff.p'.format(args.input_type, args.dir), type = 'time')
    f.feature_extract(x, ff.waveletT_RAM, 'W_RAM', '{}_feature_pickle/{}/W_RAM.p'.format(args.input_type, args.dir), type = 'time')
    
    #-------------------entropy domain----------------------------

    #f.feature_extract(x, RCMSE.RC_composite_multiscale_entropy, 'RCMSE', 
    #'{}_feature_pickle/{}/RCMSE_11_31.p'.format(args.input_type, args.output), type = 'new')
    #f.feature_extract(x, per.refined_composite_multiscale_permutation_entropy, 
    #'RCMPE', '{}_feature_pickle/{}/RCMPE_21_23_25_27_29.p'.format(args.input_type, args.output), type = 'new')
    #f.feature_extract(x, dis.refined_composite_multiscale_dispersion_entropy, 
    #'MDE', '{}_feature_pickle/{}/MDE_4class.p'.format(args.input_type, args.output), type = 'new')
    #f.feature_extract(x,wh.WSE, 
    #'WSE', '{}_feature_pickle/{}/WSE.p'.format(args.input_type, args.output), type = 'new')
    #f.feature_extract(x,wh.HSE, 
    #'HSE', '{}_feature_pickle/{}/HSE.p'.format(args.input_type, args.output), type = 'new')    

    #-------------------complexity domain----------------------------

    #f.feature_extract(x, tf.hoc, 'T_hoc', '{}_feature_pickle/{}/T_hoc.p'.format(args.input_type, args.dir), type = 'time')
    f.feature_extract(x, tf.nsi, 'T_nsi', '{}_feature_pickle/{}/T_nsi.p'.format(args.input_type, args.dir), type = 'time')
    f.feature_extract(x, tf.hjcomp, 'T_hjcomp', '{}_feature_pickle/{}/T_hjcomp.p'.format(args.input_type, args.dir), type = 'time')
    #f.feature_extract(x,FD.katz_fd, 
    #'KFD', '{}_feature_pickle/{}/KFD.p'.format(args.input_type, args.output), type = 'new')    
    #.feature_extract(x,FD.petrosian_fd, 
    #'PFD', '{}_feature_pickle/{}/PFD.p'.format(args.input_type, args.output), type = 'new')    
    #f.feature_extract(x,FD.higuchi_fd, 
    #'HFD', '{}_feature_pickle/{}/HFD.p'.format(args.input_type, args.output), type = 'new')    
    #f.feature_extract(x,morph.open_pattern_spectrum, 
    #'OPEN', '{}_feature_pickle/{}/OPEN.p'.format(args.input_type, args.output), type = 'new')    
    #f.feature_extract(x,morph.close_pattern_spectrum, 
    #'CLOSE', '{}_feature_pickle/{}/CLOSE.p'.format(args.input_type, args.output), type = 'new')
    #f.feature_extract(x,morph.curve_length, 
    #'CURVE', '{}_feature_pickle/{}/CURVE.p'.format(args.input_type, args.output), type = 'new')
    #f.feature_extract(x,morph.num_of_peaks, 
    #'PEAK', '{}_feature_pickle/{}/PEAK.p'.format(args.input_type, args.output), type = 'new')
    '''
    f.feature_extract(x,morph.avg_nonlinear_energy, 
    'NONLINEAR_ENG', '{}_feature_pickle/{}/NONLINEAR_ENG.p'.format(args.input_type, args.output), type = 'new')
    