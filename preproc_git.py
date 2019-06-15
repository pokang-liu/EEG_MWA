from scipy.io import loadmat
import utils
import os
from utils import read_mrk, read_mnt, read_cnt
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, find_peaks
import argparse


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

def read_whole_data(dataset_path):
    dir_list = os.listdir(dataset_path)
    dir_list.sort()
    x_data = []
    for directory in dir_list: # process every subject directory
        print('directory:',directory)
        files = os.listdir(os.path.join(dataset_path, directory))
        for file in files: # process every subject file
            path = os.path.join(dataset_path, directory, file)
            print("file:",file)
            if(file[0:9] == 'cnt_nback'): # EEG signal (cnt_nback)
                x = read_cnt(path)
            elif(file[0:9] == 'mrk_nback'): # read label related data, mrk_nback_event = mrk_nback.y
                time, event, y = read_mrk(path)
        print(directory)

        x_data.append(get_whole_back_data(x, time, event)) # people
    return x_data # people(26) x type (3 backs) x trials (9)


def read_seg_data(dataset_path):
    dir_list = os.listdir(dataset_path)
    dir_list.sort()
    x_data = []
    for directory in dir_list: # process every subject directory
        files = os.listdir(os.path.join(dataset_path, directory))
        for file in files: # process every subject file
            path = os.path.join(dataset_path, directory, file)
            if(file[0:9] == 'cnt_nback'): # EEG signal (cnt_nback)
                x = read_cnt(path)
            elif(file[0:9] == 'mrk_nback'): # read label related data, mrk_nback_event = mrk_nback.y
                time, event, y = read_mrk(path)
        print(directory)

        x_data.append(get_seg_back_data(x, time, event)) # people
    #print(len(x_data))#26
    #print(len(x_data[0]))#3
    #print(len(x_data[0][0]))#171
    #print(len(x_data[0][0][0]))#4xx
    #print(len(x_data[0][0][0][0]))#30
    a = []
    for i in range(len(x_data)):
        for j in range(len(x_data[i])):
            for k in range(len(x_data[i][j])):
                (a.append(len(x_data[i][j][k])))
    return x_data # people(26) x type (3 backs) x trials (171=19*9) x non-deterministic(sample point length, min = 419) x 30(# of EEG channel)
def get_seg_back_data(x, time, event):
    '''
        event: len = 567    
            112: 0-back session (total 9 trials)
            128: 2-back session (total 9 trials)
            144: 3-back session (total 9 trials)
        return: people(26) x type (3 backs) x trials (bigger than 9) x non-deterministic(sample point length) x 30(# of EEG channel)
    '''
    n_back_labels = [112, 128, 144]
    x = x[30*200:] # remove first 30 second data to align input and output length
    x_n_back = [[], [], []]


    for i, n_back_label in enumerate(n_back_labels): 
        event_idx = np.where(event == n_back_label)[0] # beginning of each session (0, 2, 3 backs) index, index dif = 21
        x_tmp = []

        for idx in event_idx:
            o_idx = idx + 1
            for trial in range(19):

                s_idx = o_idx
                e_idx = o_idx + 1

                o_idx += 1

                s_time = time[s_idx]
                e_time = time[e_idx]
    
                s_sp_idx = int(np.floor(s_time / 1000 * 200)) # sample point index 
                e_sp_idx = int(np.floor(e_time / 1000 * 200)) # sample point index 
                x_n_back[i].append(x[s_sp_idx:e_sp_idx])
    
    return x_n_back

##################################################################
def get_whole_back_data(x, time, event):
    print("get_whole_back_data")
    print("x shape",x.shape)
    '''
        remove the non-event (not 0 back, 2 back, 3 back )data

        event is another representation of y
        y is one hot encode of event
        time is the start time of the event

        event: len = 567    
            112: 0-back session (total 9 trials)
            128: 2-back session (total 9 trials)
            144: 3-back session (total 9 trials)
    '''
    n_back_labels = [112, 128, 144]
    x = x[30*200:] # remove first 30 second data to align input and output length
    x_n_back = [[], [], []]
    for i, n_back_label in enumerate(n_back_labels):
        event_idx = np.where(event == n_back_label)[0] # beginning of each session (0, 2, 3 backs) index
        x_tmp = []
        for idx in event_idx:
            s_idx = idx + 1 # start index
            e_idx = idx + 20 # end index

            s_time = time[s_idx]
            e_time = time[e_idx]

            s_sp_idx = int(np.floor(s_time / 1000 * 200)) # sample point index 
            e_sp_idx = int(np.floor(e_time / 1000 * 200)) # sample point index 

            x_n_back[i].append(x[s_sp_idx:e_sp_idx])
    print("x_n_back -1 shape",len(x_n_back[0][0]))
    print("x_n_back shape",len(x_n_back))
    

    return x_n_back # python list : 3(difficulties)  x 9 (trails) x 8000~(points)
##################################################################

def cut_whole_data(x):
    '''
        3 session(3 series of 0-, 2-, 3-back, i.e., 0->2->3->2->3->0->3->0->2)
        x shape = 26(subjects)x 3(n_backs->0,2,3) x 9(9 trial for each n back exp.) x non-deterministic(sample point length) x 30(# of EEG channel)
    '''
    # min len of signal: 8103
    EEG_signals_len = []

    x_subject = []
    y_subject = []
    label_type = [2,0,1] # (2) 0-back, (0) 2-back, (1) 3-back

    len_list = []
    for subject in range(len(x)): #26
        x_task = []
        y_task = [1,0,2,0,2,1,2,1,0,0,2,1,2,1,0,1,0,2,2,1,0,1,0,2,0,2,1] # correct task order!!!
        for s_label, task in enumerate(range(len(x[subject]))):#3
            #x_series = []
            #y_series = []
            
            for series in range(len(x[subject][task])):
                x_raw = np.transpose(x[subject][task][series]) # 30 x # sample points
                x_filtered = bandpass(x_raw) # 30 x 8436
            
                #--------------------------remove peaks of eye blinks-----------------------------#
                removed_idx = []
                blink_range = 60
                peaks, _ = find_peaks(x_filtered[-1], distance = 150, height = np.mean(x_filtered[-1]) + np.std(x_filtered[-1]))
                

                for i in range(len(peaks)):
                    s_idx = int(peaks[i] - blink_range / 2) if int(peaks[i] - blink_range / 2) > 0 else 0
                    e_idx = int(peaks[i] + blink_range / 2) if int(peaks[i] + blink_range / 2) < len(x_filtered[-1]) else 0
                    x_axis = np.arange(int(peaks[i] - blink_range / 2) , int(peaks[i] + blink_range / 2))
                    removed_idx.append(x_axis)
                    y_axis = x_filtered[0][int(peaks[i] - blink_range / 2) : int(peaks[i] + blink_range / 2)]

                removed_idx = np.array(removed_idx).reshape(-1)  
                #print(removed_idx)
                x_clean = np.delete(x_filtered, removed_idx, axis = 1)
                len_list.append(x_clean.shape[1])
                print('len_list',len_list)
                #print(x_clean.shape) #(30, 8xxx) 

                '''
                
                x_clean = artifact_reject_recursive(x_filtered) # 30 x len
                x_clean = x_clean[:, -6080:]
                '''
                x_task.append(x_clean[:, -5617:])
                # print("len x task",len(x_task)) # accumilate to 27
                # finally x_task would be (27, 30, 8xxx)               
                #y_task.append(label_type[s_label])
            
        x_subject.append(x_task) # accumulate to 26
        y_subject.append(y_task)
    print('np.min(len_list):',np.min(len_list))
    #print(len_list)
    #return np.array(x_subject), np.array(y_subject)
    return np.array(x_subject), np.array(y_subject) # x: 26 subjects x 27 tasks x 5000 points,   y: 26 subjects x 27 tasks
##################################################################

def bandpass(x):    
    fs = 200.0
    lowcut = 4.0
    highcut = 45.0

    x_f = []
    for channel in range(len(x)):
        x_f.append(butter_bandpass_filter(x[channel], lowcut, highcut, fs, order= 2))
    return np.array(x_f)

def artifact_reject(x, mul):
    check_sample = 20
    EEG_sum = np.mean(x, axis = 0) # can use only frontal electrode!!!!

    power_list = []
    for i in range(int(len(EEG_sum)/check_sample)):
        #power = np.mean(np.sqrt(np.square(EEG_sum[i*check_sample:(i+1)*check_sample])))
        segment = np.square(EEG_sum[i*check_sample:(i+1)*check_sample])
        segment.sort()
        segment = segment[-5:]
        power = np.mean(segment)
        power_list.append(power)

    threshold = np.mean(power_list) + np.std(power_list) * mul
    #print('Remain signal occupy {:.2f}% of orignal signal (mean: {:.2f}, std: {:.2f})'.\
    #    format(np.sum(np.array(power_list) < threshold) / len(power_list) * 100, np.mean(power_list), np.std(power_list)))
    
    delete_idx = []
    x_clean = []
    c = 0
    for i in range(int(len(EEG_sum)/check_sample)):
        #power = np.mean(np.sqrt(np.square(EEG_sum[i*check_sample:(i+1)*check_sample])))
        segment = np.square(EEG_sum[i*check_sample:(i+1)*check_sample])
        segment.sort()
        segment = segment[-1:]
        power = np.mean(segment)
        if(power < threshold): # remained signal
            x_clean = x[:, i*check_sample:(i+1)*check_sample] if c == 0 else np.concatenate((x_clean, x[:, i*check_sample:(i+1)*check_sample]),axis = 1)
            c += 1
        else:
            delete_idx.append(i)
    
    offset = 0

    for i in range(int(len(EEG_sum)/check_sample)):
        y = np.arange(0+offset,check_sample+offset)
        offset+=check_sample

        plt.plot(y,x[0,i*check_sample:(i+1)*check_sample], 'b')
    plt.ylim(ymax = 150,ymin=-150)#(y.min()-0.2, y.max()+0.2)
    plt.show()
    offset = 0

    for i in range(int(len(EEG_sum)/check_sample)):
        y = np.arange(0+offset,check_sample+offset)
        offset+=check_sample
        if(i in delete_idx):
            plt.plot(y,x[0,i*check_sample:(i+1)*check_sample], 'r')
        else:
            plt.plot(y,x[0,i*check_sample:(i+1)*check_sample], 'b')
    plt.ylim(ymax = 150,ymin=-150)#(y.min()-0.2, y.max()+0.2)
    plt.show()
    offset = 0

    for i in range(int(len(EEG_sum)/check_sample)):

        if(i in delete_idx):
            pass
        else:
            y = np.arange(0+offset,check_sample+offset)
            offset+=check_sample
            plt.plot(y,x[0,i*check_sample:(i+1)*check_sample], 'b')
    

    plt.ylim(ymax = 150,ymin=-150)#(y.min()-0.2, y.max()+0.2)
    plt.show()
    
    #plt.subplot(2,1,1)
    #plt.plot(x[0])
    #plt.subplot(2,1,2)
    #plt.plot(x_clean[0])
    #plt.show()
    return x_clean, len(x_clean[0])/len(x[0])
def artifact_reject_recursive(x):
    '''
        Remain percent must 75~85% is good
    '''
    #print(np.mean(x[0]))
    #plt.plot(x[0])
    #plt.show()
    x_org = x
    mul = 2
    remain_percent = 1
    #for i in range(len(mul)):
    i = 0
    use_pre_remain = False

    while(remain_percent > 0.83):
        pre_x = x
        pre_remain = remain_percent

        x, remain = artifact_reject(x, mul)
        remain_percent = remain_percent * remain

        #print(remain_percent)
        i += 1
        if(i > 50):
            break

        if(remain_percent < 0.75):
            x = pre_x
            #remain_percent = pre_remain
            if(pre_remain > 0.85):
                remain_percent = pre_remain
                mul = mul * 1.05
            else:
                use_pre_remain = True
    if(use_pre_remain == True):
        remain_percent = pre_remain
    
    print('Remain: {}%'.format(remain_percent))
    return x

def cal_min_len(x):
    '''
        x : 26 x 27 x 30 x 6400
    '''
    len_list = []
    for i in range(len(x)):
        for j in range(len(x[i])):
            for k in range(len(x[i][j])):
                len_list.append(len(x[i][j][k]))
    print(np.min(len_list))
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Preprocessing')

    parser.add_argument('-input', '--input', help = 'input directory name')
    parser.add_argument('-output', '--output', help = 'output file name')

    args = parser.parse_args()
    
    
    #parser = argparse.ArgumentParser(description = 'MW-back preprocessing')
    #parser.add_argument('-s', '--from_scratch', help = 'Need ')
    #args = parser.parse_args()

    '''
    28 EEG and 2 EoG channels name
        ['Fp1', 'AFF5h', 'AFz', 'F1', 'FC5', 
        'FC1', 'T7', 'C3', 'Cz', 'CP5', 'CP1', 
        'P7', 'P3', 'Pz', 'POz', 'O1', 'Fp2', 
        'AFF6h', 'F2', 'FC2', 'FC6', 'C4', 'T8', 
        'CP2', 'CP6', 'P4', 'P8', 'O2', 'HEOG', 'VEOG']
    '''

    '''
    a = read_mnt('EEG_01-26_MATLAB/VP001-EEG/mnt_nback.mat')
    print(a[2])
    print(a[2].shape)
    with open('saved_pickle/channel_pos.p','wb') as f:
        pickle.dump(np.transpose(a[2][0:28]), f) #x shape = 26(subjects)x 3(n_backs) x 9(9 trial for each n back exp.) x non-deterministic(sample point length) x 30(# of EEG channel)
    '''
    x = read_whole_data('{}'.format(args.input)) # 26(people) x 3(types of n-back) x 9 (trials each n-back) x non-deterministic(sample point length) x 30(# of EEG channel)
    print('xshape:',len(x))
    x_clean, y = cut_whole_data(x) # clean means that after bandpass & eye artifact removal
    dic = {}
    dic['x'] = x_clean
    dic['y'] = y
    with open('saved_pickle/{}.p'.format(args.output),'wb') as f:
        pickle.dump(dic, f) #x shape = 26(subjects)x 3(n_backs) x 9(9 trial for each n back exp.) x non-deterministic(sample point length) x 30(# of EEG channel)
    
    print('x_clean.shape', x_clean.shape) # (26, 27, 30, 5167)
    print('y.shape', y.shape) # (26, 27)
    print('y:',y )
   





























    #with open('saved_pickle/before_cut.p','wb') as f:
    #    pickle.dump(x, f) #x shape = 26(subjects)x 3(n_backs) x 9(9 trial for each n back exp.) x non-deterministic(sample point length) x 30(# of EEG channel)
    
    '''
    with open('saved_pickle/before_cut.p', 'rb') as f:
        x = pickle.load(f) 
    x, y = cut_data(x)
    dic = {}
    dic['x'] = x
    dic['y'] = y
    '''

    #with open('saved_pickle/after_cut_no_remove_eyes.p','wb') as f:
    #    pickle.dump(dic, f) #x shape = 26(subjects)x 27 x 30 x data len
    

    #with open('saved_pickle/after_cut.p','rb') as f:
    #    dic = pickle.load(f) 
    #x = dic['x']
    #print(np.array(x).shape)  
    #cal_min_len(x)

