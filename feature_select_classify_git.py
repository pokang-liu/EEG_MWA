import FreqFeature as ff
import TimeFeature as tf
import EntroFeature as ef
import argparse
import utils as f
import numpy as np
import os.path

from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from xgboost import plot_importance, plot_tree
from sklearn.feature_selection import SelectFromModel

import mifs
from xgboost import XGBClassifier
import lightgbm as lgb
from scipy.stats import rankdata
# from SAE import SAE
import torch
import time
from scipy import signal
import pywt
import pickle


from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

'''
use the args to specify feature number, enable rfe, and other tools
modify the rfe kernel in "train" function
modify classifier in "plot_classifier_result" funcation

'''

channels_log = np.array(['Fp1', 'AFF5h', 'AFz', 'F1', 'FC5', 
                'FC1', 'T7', 'C3', 'Cz', 'CP5', 'CP1', 
                'P7', 'P3', 'Pz', 'POz', 'O1', 'Fp2', 
                'AFF6h', 'F2', 'FC2', 'FC6', 'C4', 'T8', 
                'CP2', 'CP6', 'P4', 'P8', 'O2'])# 'HEOG', 'VEOG']


def MORPH_specify_scale(feature, feature_log, wanted_scale):
    '''
    specify the scale of Opening and Closing Pattern Spectrum
    '''
    wanted_idx = []
    for i in range(len(feature_log)):
        ch = feature_log[i].split('_')[2]
        if(ch in wanted_scale):
            wanted_idx.append(i)
    return feature[:,:,wanted_idx], feature_log[wanted_idx]                
def MM_specify_scale(feature, feature_log, wanted_scale):
    '''
    specify the scale of MMSE MMPE
    '''
    wanted_idx = []
    for i in range(len(feature_log)):
        ch = feature_log[i].split('_')[-1]
        if(ch in wanted_scale):
            wanted_idx.append(i)
    return feature[:,:,wanted_idx], feature_log[wanted_idx]

def RC_specify_scale(feature, feature_log, wanted_scale):
    '''
    specify the scale of RCMSE RCMPE
    '''
    wanted_idx = []
    for i in range(len(feature_log)):
        ch = feature_log[i].split('_')[3]
        if(ch in wanted_scale):
            wanted_idx.append(i)
    return feature[:,:,wanted_idx], feature_log[wanted_idx]
def specify_scale(feature, feature_log,start,end):
    feature = feature[:,:,start:end]
    feature_log = feature_log[start:end]
    return feature, feature_log



def train(x, y, test_sub, num_feature, clf):
    x_train, x_test = np.delete(x, test_sub, axis = 0), x[test_sub] # 25
    y_train, y_test = np.delete(y, test_sub, axis = 0), y[test_sub] # 25
    #print('x_train.shape before',x_train.shape) # (25(26-1) , 18(27-9) , 45)

    if(args.fcs == True):
        f_rank = f.fs_rank(x_train, y_train, fs_type = 'fcs')
        sel_feature = f_rank < args.num_feature 
        x_train = x_train[:,:,sel_feature]
        x_test = x_test[:,sel_feature]

    x_train = x_train.reshape(x_train.shape[0] * x_train.shape[1], -1) 
    #print('x_train.shape',x_train.shape) # (25x18, 45)   45 is feature number
    y_train = y_train.reshape(y_train.shape[0] * y_train.shape[1]) # 25x18

    if(args.rfe == True):
        '''
        Generate RFE rank for each validation process, total we would have 26 RFE ranks.
        feature_set_name: give a name for your feature combination EX: HOC_RCMPE_MMPE
        test_sub: the # of RFE rank 
        '''
        feature_set_name = args.feature_set_name
        if(os.path.isfile('rfe_rank/{}_'.format(feature_set_name,test_sub))):
            with open('rfe_rank/{}_'.format(feature_set_name,test_sub),'rb') as ff:
                rank = pickle.load(ff)
        else:
        #estimator = GaussianNB()
        #XGBClassifier(max_depth = 3)
            estimator = LinearSVC()
            rank = f.rank_feature(x_train, y_train,estimator)
            with open('rfe_rank/{}/{}_'.format(feature_set_name,test_sub),'wb') as ff:
                pickle.dump(rank, ff)
        selected_feature = rank < (args.num_feature+1)
        x_train = x_train[:, selected_feature]
        x_test = x_test[:, selected_feature]
    if(args.pca == True):
        pca = PCA(n_components = args.num_feature)
        pca.fit(x_train)
        x_train = pca.transform(x_train)
        x_test = pca.transform(x_test)
    #print(x_train.shape)
    clf.fit(x_train, y_train) ##  fit here!!!!!
    train_score = clf.score(x_train, y_train)
    test_score = clf.score(x_test, y_test)
    #print('Subject {}: {:.2f}%'.format(test_sub, score * 100))
    
    return train_score, test_score

def cross_validation(x, y, clf):
    train_acc_list = []
    test_acc_list = []
    for subject in range(len(x)): # 26
        train_acc, test_acc = train(x = x, y = y, 
                    test_sub = subject, 
                    num_feature = args.num_feature, 
                    clf = clf
                    )
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
    #print(np.mean(acc_list))
    return train_acc_list, test_acc_list

def plot_classifier_result(x, y, verbose):

    names = [#"Nearest Neighbors", 
             "Linear SVM", 
             #"RBF SVM", 
             ###"Decision Tree", 
             ###"Random Forest", 
             ###"Neural Net", 
             ###"AdaBoost",
             #"Naive Bayes", 
             ###'XGBClassifier',
             ###"QDA"
             ]
    n = np.random.randint(50,500)
    lr = np.random.uniform(0.05,0.5)
    md = np.random.randint(1,7)
    mcw = np.random.randint(1,3)
    g = np.random.uniform(0,0.2)
    subs = np.random.uniform(0.6,0.8)
    col = np.random.uniform(0.6,0.8)
    pos = 1
    reg = 1
    clfs = [ # singal feature: LSVM: 0.02, RSVG: 7e-3, XGB: 500,0.15,1,1,0,0.8,0.8,1,1
            #KNeighborsClassifier(51),
            #SVC(kernel="linear", C=1)
            #SVC(gamma=5e-4, C=1),
            ###DecisionTreeClassifier(max_depth=3),
            ###RandomForestClassifier(max_depth=3, n_estimators=100, max_features=1),
            #MLPClassifier(alpha=6),
            #AdaBoostClassifier(),
            GaussianNB()
            #XGBClassifier(max_depth = 3)

            #'''
            #XGBClassifier(n_estimators=500,  # for single feature
            #             learning_rate =0.15,
            #             max_depth = 1,
            #             min_child_weight = 1,
            #             gamma = 0,
            #             subsample = 0.8,
            #             colsample_bytree = 0.8,
            ##             scale_pos_weight = 1,
           #              reg_alpha = 1),
           #3 '''
            #XGBClassifier(n_estimators=1000,  # for single feature
            #              learning_rate =0.00000005,
            #              max_depth = 5,
            #              min_child_weight = 1,
            #              gamma = 0,
            #              subsample = 0.8,
            #              colsample_bytree = 0.8,
            #              scale_pos_weight = 1,
            #              reg_alpha = 5),
            ##QuadraticDiscriminantAnalysis()
            ]

    #clf = [clf1, clf2, clf3, clf4]
    #lc = ['r', 'g', 'b', 'c']
    train_list, test_list = [], []
    for i in range(len(clfs)):
        train_acc, test_acc = cross_validation(x, y,SVC(kernel="linear", C=0.05))
        if(verbose):
            print('{:>25}: {:.2f}+/-{:.2f}%(train), {:.2f}+/-{:.2f}%(test)'.format(names[i], np.mean(train_acc)*100, np.std(train_acc)*100,np.mean(test_acc)*100,
                np.std(test_acc)*100))
        train_list.append(np.mean(train_acc))
        test_list.append(np.mean(test_acc))
    if(verbose):
        print('-----------------------------------------------------------------')
        print('{:>25}: {:.2f}%(train), {:.2f}%(test)'.format('Mean accuracy', np.mean(train_list)*100, np.mean(test_list)*100))

    return np.mean(test_list)*100
        #plt.plot(result, label = label[i], color = lc[i])
    #plt.legend()
    #plt.show()


def run_channel_acc(feature, y, log, fig_name):
    channels_log = ['Fp1', 'AFF5h', 'AFz', 'F1', 'FC5', 
                    'FC1', 'T7', 'C3', 'Cz', 'CP5', 'CP1', 
                    'P7', 'P3', 'Pz', 'POz', 'O1', 'Fp2', 
                    'AFF6h', 'F2', 'FC2', 'FC6', 'C4', 'T8', 
                    'CP2', 'CP6', 'P4', 'P8', 'O2']# 'HEOG', 'VEOG']
    feature = f.normalize(feature) # 26 x 27 x 112(4*28) normalize by person
    acc_list = []
    #fn = log[0].split('_')[1] + '_' + log[0].split('_')[2] # fn: feature_name
    fn = fig_name
    for i in range(len(channels_log)):
        want_channel = channels_log[i]
        remove_idx = []
        for j in range(len(log)):
            ch_log = log[j].split('_')[0]
            if(ch_log != want_channel):
                remove_idx.append(j)

        s_feature = np.delete(feature, remove_idx, axis = 2)
        print(s_feature.shape)
        x_data, y_data = f.take_0_3_back(s_feature, y)
        acc = plot_classifier_result(x_data,y_data, verbose = False)
        acc_list.append(acc)

    rank = rankdata(-np.array(acc_list), method = 'min')
    with open('result/{}.txt'.format(fn), 'w') as file:
        for i in range(len(rank)):
            file.write('{},{:.2f},{}\n'.format(channels_log[i], acc_list[i], rank[i]))
    
    cm = plt.cm.get_cmap('RdYlBu')
    pos = f.pickle_op('saved_pickle/channel_pos.p', 'r')
    for i in range(len(acc_list)):
        sc = plt.scatter(pos[i][0], pos[i][1], c = acc_list[i], cmap = cm, vmin=50, vmax=65, s = 500)
        plt.text(pos[i][0]-0.14, pos[i][1]+0.10, '{}_{:.0f}_{}'.format(channels_log[i], acc_list[i], rank[i]), fontsize=7)
    plt.colorbar(sc)
    #plt.title('{}'.format(fn))
    #plt.axis('off')


    #plt.savefig('paper/channel_imp/eps/{}.eps'.format(fn), format ='eps',bbox_inches='tight')
    plt.show()

    plt.clf()
def concat_feature_list(f_list, f_log_list):
    for i in range(len(f_list)):
        if(i == 0):
            arr = f_list[i]
            arr_log = f_log_list[i]
        else:
            arr = np.concatenate((arr, f_list[i]), axis = 2)
            arr_log = np.concatenate((arr_log, f_log_list[i]))
    return arr, arr_log
def xgb_feature_select(x, y, log):
    x_train = x.reshape(-1, x_data.shape[2])
    y_train = y.reshape(-1)
    clf =    XGBClassifier(n_estimators=100,
               learning_rate =0.05,
               max_depth = 5,
               min_child_weight = 2,
               gamma = 0,
               subsample = 0.8,
               colsample_bytree = 0.8,
               scale_pos_weight = 1,
               reg_alpha = 1) 

# make predictions for test data and evaluate

  
    
    import xgboost
    clf.fit(x_train, y_train)
    print(clf.feature_importances_)
    print(clf.feature_importances_.shape)
    arg_idx = np.argsort(-clf.feature_importances_)<80
    x = x[:,:,arg_idx]
    print(x.shape)
    plot_classifier_result(x,y, verbose = True)
    '''
    #idx = (np.argsort(-clf.feature_importances_)) # small -> important
    #log_sort = (feature_log[idx])
    #accum = [0] * 28
    #for i in range(100):
    #    ch = log_sort[i].split('_')[0]
    #    ch_idx = np.where(ch == channels_log)[0][0]
    #    accum[ch_idx] = accum[ch_idx] + 1
#
    #ch_rank = rankdata(-np.array(accum), method = 'min')
    #cm = plt.cm.get_cmap('RdYlBu')
    #pos = f.pickle_op('saved_pickle/channel_pos.p', 'r')
    #for i in range(len(accum)):
    #    sc = plt.scatter(pos[i][0], pos[i][1], c = accum[i], cmap = cm, vmin=np.min(accum), vmax=np.max(accum), s = 500)
    #    plt.text(pos[i][0]-0.05, pos[i][1], '{}_{}_{}'.format(channels_log[i], accum[i], ch_rank[i]), fontsize=7)
    #plt.colorbar(sc)
    #plt.show()
    plot_tree(clf, num_trees = 4)
    fscore = clf.get_score()
    #print(fscore)
    plot_importance(clf, max_num_features = 10)
    plt.show()
    '''
def plot_singal_channel_acc(feature_list, feature_log_list,y, fig_name):
    feature = f.normalize(feature_list)
    x_data, y_data = f.take_0_3_back(feature, y)
    run_channel_acc(x_data, y_data, feature_log_list, fig_name)
    
    #for i in range(len(feature_list)):
    #    print(i)
    #    feature = f.normalize(feature_list[i])
    #    x_data, y_data = f.take_0_3_back(feature, y)
    #    run_channel_acc(x_data, y_data, feature_log_list[i], fig_name)



def RFE_normalize(x_data,y_data):
# map features to [-1, 1]
    #for i in range(x_data.shape[1]):
    x_data=np.transpose(x_data, (2, 1, 0))
    print('x_data.shape 1: ', x_data.shape) # (45, 18, 2)
    x_data=np.reshape(x_data, (x_data.shape[0],x_data.shape[1]*x_data.shape[2]))
    print('x_data.shape 2: ', x_data.shape) # (45, 36)

    x_data_max = np.max(x_data, axis=1)
    x_data_min = np.min(x_data, axis=1)
    print('x_data max shape: ',x_data_max.shape)
    x_data = np.transpose(x_data) # (36, 45)
    print('x_data.shape 3: ', x_data.shape) #

    x_data = (x_data - x_data_min) / (x_data_max - x_data_min)
    x_data = x_data * 2 - 1
    print('x_data.shape 3: ', x_data.shape) #
    #print('x_data normalizeation: ',x_data)

    x_data = np.transpose(x_data) # transpose back (45, 36)
    print('x_data.shape 4: ', x_data.shape) #

    x_data=np.reshape(x_data, (x_data.shape[0],27,int(x_data.shape[1]/27)))
    print('x_data.shape 5: ', x_data.shape) #

    x_data=np.transpose(x_data, (2, 1, 0))

    print('x_data final: ',x_data.shape)
    return x_data, y_data
def normalize(x_data,y_data):
# map features to [-1, 1]
    #for i in range(x_data.shape[1]):
    x_data=np.transpose(x_data, (2, 1, 0))
    print('x_data.shape 1: ', x_data.shape) # (45, 18, 2)
    x_data=np.reshape(x_data, (x_data.shape[0],x_data.shape[1]*x_data.shape[2]))
    print('x_data.shape 2: ', x_data.shape) # (45, 36)

    x_data_max = np.max(x_data, axis=1)
    x_data_min = np.min(x_data, axis=1)
    print('x_data max shape: ',x_data_max.shape)
    x_data = np.transpose(x_data) # (36, 45)
    print('x_data.shape 3: ', x_data.shape) #

    x_data = (x_data - x_data_min) / (x_data_max - x_data_min)
    x_data = x_data * 2 - 1
    print('x_data.shape 3: ', x_data.shape) #
    #print('x_data normalizeation: ',x_data)

    x_data = np.transpose(x_data) # transpose back (45, 36)
    print('x_data.shape 4: ', x_data.shape) #

    x_data=np.reshape(x_data, (x_data.shape[0],18,int(x_data.shape[1]/18)))
    print('x_data.shape 5: ', x_data.shape) #

    x_data=np.transpose(x_data, (2, 1, 0))

    print('x_data final: ',x_data.shape)
    return x_data, y_data 


    
if __name__ == '__main__':
    
    
    '''
    remember to modify the default value of the arguments^O^ 
    '''
    parser = argparse.ArgumentParser(description = 'Extract CNN feature')
    #parser.add_argument('-t', '--input_type', help = 'input file type whole or seg', default = 'whole', choices = ['whole', 'seg'])
    parser.add_argument('-rfe', '--rfe', help = 'use rfe feature selection or not, -s = True', action = 'store_true')
    parser.add_argument('-feature_set_name', type = str, help = 'feature_set_name',default = 'my_feature_set')
    parser.add_argument('-fcs', '--fcs', help = 'use fcs feature selection or not, -s = True', action = 'store_true')
    parser.add_argument('-nf', '--num_feature', help = 'number of feature selected', default = 20, type = int)
    #parser.add_argument('-pca', '--pca', help = 'whether use PCA or not', action = 'store_true')
    #parser.add_argument('-wt', '--wavetype', type = int)
    parser.add_argument('-dir', '--dir', help = 'feature directory', default = '26eog')
    #parser.add_argument('-c', '--svmC', help = 'svm param', default = 0.05, type = float)
    parser.add_argument('-input', '--input', help = 'input file to read y, default = whole_trial_2_eog',default = '26_eog_v2')
    args = parser.parse_args()
    


    #-------------------------------   read lebel (y) -------------------------------

    dic = f.pickle_op(file_name = 'saved_pickle/{}.p'.format(args.input), mode = 'r')
    y = dic['y'] # 26 x 27
    x = dic['x'] # 26 x 27 x 30 x 6080 or 26 x 513 x 30 x 419
    x = f.sub_baseline(x)
    '''
    ff.wavelet_power(x[0][0][0])
    f, t, Zxx = signal.stft(x[0][0][0],200)
    print(f.shape)
    print(t.shape)
    '''
    #-------------------------------   read feature -------------------------------
    # -------------------------------uncomment the feature you want-------------------------------
    
    FT_psd, FT_psd_log = f.read_feature('{}_feature_pickle/{}/FT_psd.p'.format(args.input_type, args.dir))
    T_mean_power, T_mean_power_log = f.read_feature('{}_feature_pickle/{}/T_mean_power.p'.format(args.input_type, args.dir))
    T_mean, T_mean_log = f.read_feature('{}_feature_pickle/{}/T_mean.p'.format(args.input_type, args.dir))
    T_std, T_std_log = f.read_feature('{}_feature_pickle/{}/T_std.p'.format(args.input_type, args.dir))
    T_first_diff, T_first_diff_log = f.read_feature('{}_feature_pickle/{}/T_first_diff.p'.format(args.input_type, args.dir))
    T_second_diff, T_second_diff_log = f.read_feature('{}_feature_pickle/{}/T_second_diff.p'.format(args.input_type, args.dir))
    STFT_power, STFT_power_log = f.read_feature('{}_feature_pickle/{}/STFT_power.p'.format(args.input_type, args.dir))
    FT_se, FT_se_log = f.read_feature('{}_feature_pickle/{}/FT_se.p'.format(args.input_type, args.dir))
    T_hoc, T_hoc_log = f.read_feature('{}_feature_pickle/{}/T_hoc.p'.format(args.input_type, args.dir))
    T_pfd, T_pfd_log = f.read_feature('{}_feature_pickle/{}/T_pfd.p'.format(args.input_type, args.dir))
    T_nsi, T_nsi_log = f.read_feature('{}_feature_pickle/{}/T_nsi.p'.format(args.input_type, args.dir))
    T_hjcomp, T_hjcomp_log = f.read_feature('{}_feature_pickle/{}/T_hjcomp.p'.format(args.input_type, args.dir))
    #T_wavelet, T_wavelet_log = f.read_feature('{}_feature_pickle/{}/T_wavelet.p'.format(args.input_type, args.dir))
    F_wavelet, F_wavelet_log = f.read_feature('{}_feature_pickle/{}/F_wavelet.p'.format(args.input_type, args.dir))
    EN_mse, EN_mse_log = f.read_feature('{}_feature_pickle/{}/EN_mse.p'.format(args.input_type, args.dir))
    
    
    print('FT_psd_log shape',FT_psd_log.shape)
    print('T_mean_power_log shape',T_mean_power_log.shape)
    print('T_mean_log shape',T_mean_log.shape)
    print('T_std_log shape',T_std_log.shape)
    print('T_first_diff_log shape',T_first_diff_log.shape)
    print('T_second_diff_log shape',T_second_diff_log.shape)
    print('STFT_power_log shape',STFT_power_log.shape)
    print('FT_se_log shape',FT_se_log.shape)
    print('T_hoc_log shape',T_hoc_log.shape)
    print('T_pfd_log shape',T_pfd_log.shape)
    print('T_nsi_log shape',T_nsi_log.shape)
    print('T_hjcomp_log shape',T_hjcomp_log.shape)
    print('F_wavelet_log shape',F_wavelet_log.shape)
    print('EN_mse_log shape',EN_mse_log.shape)


    W_MAV, W_MAV_log = f.read_feature('{}_feature_pickle/{}/W_MAV.p'.format(args.input_type, args.dir))
    W_AP, W_AP_log = f.read_feature('{}_feature_pickle/{}/W_AP.p'.format(args.input_type, args.dir))
    W_SD, W_SD_log = f.read_feature('{}_feature_pickle/{}/W_SD.p'.format(args.input_type, args.dir))
    W_first_diff, W_first_diff_log = f.read_feature('{}_feature_pickle/{}/W_first_diff.p'.format(args.input_type, args.dir))
    W_second_diff, W_second_diff_log = f.read_feature('{}_feature_pickle/{}/W_second_diff.p'.format(args.input_type, args.dir))
    W_RAM, W_RAM_log = f.read_feature('{}_feature_pickle/{}/W_RAM.p'.format(args.input_type, args.dir))
    print('W_MAV_log shape',W_MAV_log.shape)
    print('W_AP_log shape',W_AP_log.shape)
    print('W_SD_log shape',W_SD_log.shape)
    print('W_first_diff_log shape',W_first_diff_log.shape)
    print('W_second_diff_log shape',W_second_diff_log.shape)
    print('W_RAM_log shape',W_RAM_log.shape)

    psdMAV, psdMAV_log = f.read_feature('{}_feature_pickle/{}/F_psdMAV.p'.format(args.input_type, args.dir))
    print('psdMAV_log shape',psdMAV_log.shape)
    
    #-------------------------------    RCMSE -------------------------------
    RCMSE, RCMSE_log = f.read_feature('{}_feature_pickle/{}/RCMSE_all.p'.format(args.input_type, args.dir))
    #RCMSE2, RCMSE_log2 = f.read_feature('{}_feature_pickle/{}/RCMSE_20_25.p'.format(args.input_type, args.dir))
    #RCMSE, RCMSE_log =f.specify_channel(RCMSE, RCMSE_log, wanted_channel = ['P7'])
    print('RCMSE shape',RCMSE_log.shape)

    #-------------------------------    RCMSE -------------------------------
    RCMPE1, RCMPE_log1 = f.read_feature('{}_feature_pickle/{}/RCMPE_11_13_15_17_19.p'.format(args.input_type, args.dir))
    RCMPE2, RCMPE_log2 = f.read_feature('{}_feature_pickle/{}/RCMPE_21_23_25_27_29.p'.format(args.input_type, args.dir))
    
    

    #-------------------------------    You can specify RCMSE scale here -------------------------------
    #RCMPE2, RCMPE_log2 = RC_specify_scale(RCMPE2,RCMPE_log2,wanted_scale = ['s21','s23'])

    #print('RCMPE222222 shape',RCMPE2.shape) # 420 = s21 23 25 27 29 x m 345 x channel 28



    #-------------------------------    Fractal Dimension -------------------------------
    KFD, KFD_log = f.read_feature('{}_feature_pickle/{}/KFD.p'.format(args.input_type, args.dir))
    PFD, PFD_log = f.read_feature('{}_feature_pickle/{}/PFD.p'.format(args.input_type, args.dir)) 
    HFD, HFD_log = f.read_feature('{}_feature_pickle/{}/HFD.p'.format(args.input_type, args.dir)) 
    print('KFD',KFD_log.shape)
    print('PFD',PFD_log.shape)
    print('HFD',HFD_log.shape)
    
    #-------------------------------    WSE HSE -------------------------------
    WSE, WSE_log = f.read_feature('{}_feature_pickle/{}/WSE.p'.format(args.input_type, args.dir))
    HSE, HSE_log = f.read_feature('{}_feature_pickle/{}/HSE.p'.format(args.input_type, args.dir))   
    print('WSE',WSE_log.shape)
    print('HSE',HSE_log.shape)
    
    #------------------------------- MMPE -------------------------------
    MMPE_AF, MMPE_log_AF = f.read_feature('{}_feature_pickle/{}/MMPE_AFF5h_AFF6h_AFz.p'.format(args.input_type, args.dir))
    MMPE_F, MMPE_log_F = f.read_feature('{}_feature_pickle/{}/MMPE_F1_F2.p'.format(args.input_type, args.dir))
    MMPE_FC, MMPE_log_FC = f.read_feature('{}_feature_pickle/{}/MMPE_FC1_FC2.p'.format(args.input_type, args.dir))
    MMPE_CP, MMPE_log_CP = f.read_feature('{}_feature_pickle/{}/MMPE_CP1_CP2.p'.format(args.input_type, args.dir))
    MMPE_P, MMPE_log_P = f.read_feature('{}_feature_pickle/{}/MMPE_P3_P4_Pz.p'.format(args.input_type, args.dir))
    MMPE_C, MMPE_log_C = f.read_feature('{}_feature_pickle/{}/MMPE_C3_C4_Cz.p'.format(args.input_type, args.dir))
    print('MMPE_C',MMPE_C.shape)
    print('MMPE_P',MMPE_P.shape)
    print('MMPE_P',MMPE_log_P.shape)
    

    #-------------------------------    You can specify MMPE scale here -------------------------------
    #scale_start = 15
    #scale_end = 21
    #MMPE1, MMPE_log1 = MM_specify_scale(MMPE1,MMPE_log1,scale_start-11,scale_end-11)
    #MMPE2, MMPE_log2 = MM_specify_scale(MMPE2,MMPE_log2,scale_start-11,scale_end-11)



    #------------------------------- MMSE -------------------------------
    MMSE_AF, MMSE_log_AF = f.read_feature('{}_feature_pickle/{}/MMSE_AFF5h_AFF6h_AFz.p'.format(args.input_type, args.dir))
    MMSE_F, MMSE_log_F = f.read_feature('{}_feature_pickle/{}/MMSE_F1_F2.p'.format(args.input_type, args.dir))
    MMSE_FC, MMSE_log_FC = f.read_feature('{}_feature_pickle/{}/MMSE_FC1_FC2.p'.format(args.input_type, args.dir))
    MMSE_CP, MMSE_log_CP = f.read_feature('{}_feature_pickle/{}/MMSE_CP1_CP2.p'.format(args.input_type, args.dir))
    MMSE_P, MMSE_log_P = f.read_feature('{}_feature_pickle/{}/MMSE_P3_P4_Pz.p'.format(args.input_type, args.dir))
    MMSE_C, MMSE_log_C = f.read_feature('{}_feature_pickle/{}/MMSE_C3_C4_Cz.p'.format(args.input_type, args.dir))
    MMSE_O, MMSE_log_O = f.read_feature('{}_feature_pickle/{}/MMSE_O1_O2.p'.format(args.input_type, args.dir))
    print('MMSE_log_O',MMSE_log_O.shape)
    print('MMSE_log_P',MMSE_log_P.shape)

    #-------------------------------    You can specify MMSE scale here -------------------------------
    #scale_start = 15
    #scale_end = 21
    #MMSE_AF, MMSE_log_AF = MM_specify_scale(MMSE_AF,MMSE_log_AF,scale_start-11,scale_end-11)
    #------------------------------- Pattern Spectrum -------------------------------

    OPEN, OPEN_log = f.read_feature('{}_feature_pickle/{}/OPEN.p'.format(args.input_type, args.dir))
    CLOSE, CLOSE_log = f.read_feature('{}_feature_pickle/{}/CLOSE.p'.format(args.input_type, args.dir))
    print('OPEN_log',OPEN_log)
    print('CLOSE_log',CLOSE_log)


    #-------------------------------    You can specify Pattern Spectrum scale here -------------------------------
    #OPEN, OPEN_log = MORPH_specify_scale(OPEN,OPEN_log,wanted_scale=['s1'])
    #CLOSE, CLOSE_log = MORPH_specify_scale(CLOSE,CLOSE_log,wanted_scale=['s1'])



    #------------------------------- Other complexity feature-------------------------------
    CURVE, CURVE_log = f.read_feature('{}_feature_pickle/{}/CURVE.p'.format(args.input_type, args.dir))
    PEAK, PEAK_log = f.read_feature('{}_feature_pickle/{}/PEAK.p'.format(args.input_type, args.dir))
    ENG, ENG_log = f.read_feature('{}_feature_pickle/{}/NONLINEAR_ENG.p'.format(args.input_type, args.dir))

    print('CURVE_log',CURVE_log.shape)
      
    
    rem_idx = []
    for i in range(len(T_hoc[0][0])):
        if(i % 10!= 8):
            rem_idx.append(i)
    T_hoc8 = np.delete(T_hoc, rem_idx, axis = 2)
    T_hoc8_log = np.delete(T_hoc_log, rem_idx)
    feature_list = []
    feature_log_list = []

    ####### Time complexity #########
    TC_feature_list =     [T_hoc,T_nsi,T_hjcomp,EN_mse]
    TC_feature_log_list = [T_hoc_log,T_nsi_log,T_hjcomp_log,EN_mse_log]#, T_hoc8_log]
    ####### Time statistical #########
    TS_feature_list =     [T_mean_power,T_mean,T_std,T_first_diff,T_second_diff]
    TS_feature_log_list = [T_mean_power_log,T_mean_log,T_std_log,T_first_diff_log,T_second_diff_log]#, T_hoc8_log]
    ####### Time  #########
    T_feature_list =     [T_mean_power,T_mean,T_std,T_first_diff,T_second_diff,T_hoc,T_nsi,T_hjcomp,EN_mse]
    T_feature_log_list = [T_mean_power_log,T_mean_log,T_std_log,T_first_diff_log,T_second_diff_log,T_hoc_log,T_nsi_log,T_hjcomp_log,EN_mse_log]#, T_hoc8_log]
    ####### Frequency domain #########
    F_feature_list =     [FT_psd,FT_se]
    F_feature_log_list = [FT_psd_log, FT_se_log]#, T_hoc8_log]
    ####### Wavelet related ##########
    W_feature_list =     [W_MAV,W_AP,W_SD,W_RAM,W_first_diff,W_second_diff]
    W_feature_log_list = [W_MAV_log,W_AP_log,W_SD_log,W_RAM_log,W_first_diff_log,W_second_diff_log]
    ####### All feature #####################
    All_feature_list = [T_hoc,T_nsi,T_hjcomp,EN_mse,T_mean_power,T_mean,T_std,T_first_diff,T_second_diff,FT_psd,FT_se,W_MAV,W_AP,W_SD,W_RAM]
    All_feature_log_list = [T_hoc_log,T_nsi_log,T_hjcomp_log,EN_mse_log,T_mean_power_log,T_mean_log,T_std_log,T_first_diff_log,T_second_diff_log,
    FT_psd_log, FT_se_log,W_MAV_log,W_AP_log,W_SD_log,W_RAM_log]

    ######### RCMSE RCMPE ##############
    RCMSE_feature_list = [RCMSE]
    RCMSE_feature_log_list = [RCMSE_log]

    RCMPE_feature_list = [RCMPE1,RCMPE2]
    RCMPE_feature_log_list = [RCMPE_log1,RCMPE_log2]
    ####### mmse #############################
    MMSE_feature_list = [MMSE_AF,MMSE_F,MMSE_FC,MMSE_C,MMSE_CP,MMSE_P,MMSE_O]
    MMSE_feature_log_list = [MMSE_log_AF,MMSE_log_F,MMSE_log_FC,MMSE_log_C,MMSE_log_CP,MMSE_log_P,MMSE_log_O]
    ####### mmpe ############################
    MMPE_feature_list = [MMPE_AF,MMPE_F,MMPE_FC,MMPE_C,MMPE_CP,MMPE_P]
    MMPE_feature_log_list = [MMPE_log_AF,MMPE_log_F,MMPE_log_FC,MMPE_log_C,MMPE_log_CP,MMPE_log_P]


    ####### wse hse ##########################
    SE_feature_list = [WSE]
    SE_feature_log_list = [WSE_log]

    ####### FD ##########################
    #FD_feature_list = [KFD,PFD,HFD]
    #FD_feature_log_list = [KFD_log,PFD_log,HFD_log]
    FD_feature_list = [KFD]
    FD_feature_log_list = [KFD_log]

    ####### morphology ##########################
    MORPH_feature_list = [OPEN,CLOSE]
    MORPH_feature_log_list = [OPEN_log,CLOSE_log]




    Good_feature_list = [FT_psd,T_hoc,W_RAM,W_MAV,W_AP,W_SD ]


    WSE_list =[WSE]
    WSE_log_list = [WSE_log]


    HOC_feature_list = [T_hoc]
    HOC_feature_log_list = [T_hoc_log]

    Good_feature_log_list = [FT_psd_log,T_hoc_log,W_RAM_log,W_MAV_log,W_AP_log,W_SD_log]
    Good_feature_list = TS_feature_list +  F_feature_list + W_feature_list
    Good_feature_log_list =TS_feature_log_list + F_feature_log_list + W_feature_log_list 


    Good_feature, Good_feature_log = concat_feature_list(Good_feature_list, Good_feature_log_list)


    #------------------------------- doing channel specification here-------------------------------
    #Good_feature, Good_feature_log = f.specify_channel(Good_feature, Good_feature_log, wanted_channel = ['FC1','FC2'])#,'Fp1','AFz','Fp2','AFF5h','AFF6h'])
    #print('Good_feature.shape after specify_channel: ',Good_feature.shape)
    #print('feature log',feature_log)


    
    ######## when we need to use frontal HOC + MMPE1 ################
    Good_feature_list = [Good_feature]
    Good_feature_log_list = [Good_feature_log]
    
    ######## when we need to use frontal HOC + MMPE1 end ################   
   
    #------------------------------- MMSE MMPE should concatenate here after channel specification------------------------------
    current_feature_list = Good_feature_list #+MMPE_feature_list
    current_feature_log_list = Good_feature_log_list #+MMPE_feature_log_list
    
    
    feature_list = current_feature_list  
    feature_log_list = current_feature_log_list 
    feature, feature_log = concat_feature_list(feature_list, feature_log_list)
    print('feature_log shape',feature_log.shape)
    
    #------------------------------- for channel specification and feature domain specification analysis------------------------------

    '''
    feature_list = T_feature_list
    feature_log_list = T_feature_log_list
    feature, feature_log = concat_feature_list(feature_list, feature_log_list)
    plot_singal_channel_acc(feature, feature_log,y, fig_name = 'T')

    feature_list = F_feature_list
    feature_log_list = F_feature_log_list
    feature, feature_log = concat_feature_list(feature_list, feature_log_list)
    plot_singal_channel_acc(feature, feature_log,y, fig_name = 'F')
    
    feature_list = All_feature_list
    feature_log_list = All_feature_log_list
    feature, feature_log = concat_feature_list(feature_list, feature_log_list)
    plot_singal_channel_acc(feature, feature_log,y, fig_name = 'TF')
    '''

    #feature, feature_log = f.band_selection(feature, feature_log, 'beta')
    #print(feature_log)
    #print(feature.shape)
    #print(feature.shape)

    #-------------------------------  normalize per subject -------------------------------
    feature = f.normalize(feature) # normalize per subject




    '''
    #-------------------------------  Maximum relevance Minimum Redudancy feature selection -------------------------------
    print('mrmr gogo')
    x_train = feature.reshape(feature.shape[0] * feature.shape[1], -1)
    print(x_train.shape)
    y_train = y.reshape(y.shape[0] * y.shape[1])
    if(os.path.isfile('jmim_rank/rcmse_oldgood')):
        with open('jmim_rank/rcmse_oldgood','rb') as ff:
            rank = np.array(pickle.load(ff))
    else:
    # define MI_FS feature selection method
        print("do the slow mrmr")
        feat_selector = mifs.MutualInformationFeatureSelector(method='MRMR',n_features = 60, verbose=2, 
                                                  categorical = False)

        # find all relevant features
        feat_selector.fit(x_train, y_train)

        # check selected features
        print('feat_selector.support_', feat_selector.support_)

        # check ranking of features
        print('feat_selector.ranking_', feat_selector.ranking_)

        rank = np.array(feat_selector.ranking_)
        # call transform() on X to filter it down to selected features
        X_filtered = feat_selector.transform(x_train)

        print('hello let do the slow RFE selection job')
        with open('jmim_rank/rcmse_oldgood','wb') as ff:
            pickle.dump(rank, ff)
    #print('rank',rank)
    feature, feature_log = feature[:,:,rank], feature_log[rank]
    print('feature shape',feature.shape)
    acclist = []
    
    for i in range (5):
    #    print('i',i)
    #print('rank',rank)
    
        sel_feature = rank[:(10*(i+1))] 
        feature1, feature_log1 =  feature[:,:,sel_feature], feature_log[sel_feature]

        #feature1, feature_log1 = f.feature_select(feature, feature_log, rank, num_feature =100)
        print('feature shape after mrmr',feature1.shape)

        print('feature_log',feature_log1)
        x_data, y_data = f.take_0_3_back(feature1, y)
        acc = plot_classifier_result(x_data,y_data, verbose = True)
        acclist.append(acc)
    print('acclist',acclist)
    
    
    #####################  mrmr end #####################
    '''


    #------------------------ training here (plot_classifier_result) ------------------------------

    x_data, y_data = f.take_0_3_back(feature, y)

    acc = plot_classifier_result(x_data,y_data, verbose = True)
    acclist.append(acc)
    print('acclist',acclist)      
    #feature1, feature_log1 = f.feature_select(feature, feature_log, rank, num_feature = 40)

    #print(feature_log1[90:110])
    
    #------------------------ plot accuracy list ------------------------------
    #x = np.arange(10,210,10)
    #plt.ioff()

    #plt.xticks(np.arange(min(x), max(x), 10))

    #plt.xlabel("feature number",labelpad=1)
    #plt.plot(x,acclist,'bo-', linewidth=1,markersize=3)
    #plt.savefig('acclist.png')
    



 