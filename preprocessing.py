# -*- coding: utf-8 -*-
# 选择通道，划分时间窗       32*(40*40*8064)
# 标签进行2、3、4...分类'
# data&label保存在一个文件中，共32*a/v/4c个32*(1160*4*512)

import numpy as np
import pickle as pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
## Setting Parameters
PARTICIPANTS_NUM = 32
VIDEOS_NUM       = 40
EEG_channels= ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'FC5', 'FC6', 'FC1', 'FC2', 'AF3', 'AF4', 'C3',
              'CP5' 'CP1' 'P3' 'P7' 'PO3' 'O1' 'OZ' 'Pz' 'C4' 'T7' 'T8' 'Fz' 'Cz' 'CP6' 'CP2' 'P4' 'P8' 'PO4' 'O2']
Peripheral_channels=['GSR','HST','BVP']

#subject_list = ['01','02','03']
subject_list = [
    '01','02','03','04','05','06','07','08','09','10'
    ,'11','12','13','14','15','16','17','18','19','20'
    ,'21','22','23','24','25','26','27','28','29','30'
    ,'31','32'
]

# Setting Label
def label_mapping4c(valence, arousal):
    # HVHA, HVLA, LVHA, and LVLA

    if (valence > 5 and arousal > 5):
        label = 0  # 'HVHA'
    elif (valence > 5 and arousal <= 5):
        label = 1  # 'HVLA'
    elif (valence <= 5 and arousal > 5):
        label = 2  # 'LVHA'
    elif (valence <= 5 and arousal <= 5):
        label = 3  # 'LVLA'
    return label
def label_mappinga(valence, arousal):
    if (arousal >5):#arousal >= 4.5   valence >= 4.5
        label = 1
    else:
        label = 0
    return label
def label_mappingv(valence, arousal):
    if (valence >5):#arousal >= 4.5   valence >= 4.5
        label = 1
    else:
        label = 0
    return label

q='4c'#/v/4c
def map_label(label):
    if (q== 'a'):
        new_label = label_mappinga(label[0], label[1])
    elif (q=='v'):
        new_label = label_mappingv(label[0], label[1])
    elif (q == '4c'):
        new_label = label_mapping4c(label[0], label[1])
    return new_label

#data: (trial, channel, data) ,label: (trial,)
def epoching(sub, channel, window_size, step_size):
    signal = []
    scaler = MinMaxScaler()
    with open("/opt/data/private/deap_datas_python/s" + sub + '.dat', 'rb') as file:
        subject = pickle.load(file, encoding='latin1')  # resolve the python 2 data problem by encoding : latin1

        for i in range(0, 40):
            # loop over 0-39 trials
            data = subject["data"][i]#subject为4维，data为2维40*8064，channel*datapoint
            #print('datashapei',data.shape)#channel*datapoint，40*8064
            #print('datashape', subject['data'].shape)#(trial, channel, data)，40*40*8064
            data = np.delete(data,np.s_[0:384],axis=1)#删除3s基线数据#40*7680
            np.mean(data)#未使用
            for k in range(0, 40):
                data[k]=data[k]-np.mean(data[k])#AMR#行或列处理
                data_min=np.min(data[k])
                data_max = np.max(data[k])
                data[k] = (data[k] - data_min) / (data_max - data_min)
            '''data[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
             29, 30, 31]]=data[[0, 1, 10, 11, 2, 4, 24, 3, 5, 6, 8, 7, 9, 12, 25, 21, 13, 14, 27, 26, 16, 15, 20, 28, 29, 17,
                                30, 18, 19, 31, 22, 23]]'''
            labels = subject["labels"][i]#(4,)
            #print(labels.shape)#(4,)
            #print(subject['labels'].shape)#(40, 4)，channel*4
            label = map_label(labels)
            labels_4class = np.array(label)#()
            #print('label_shape',labels_4class.shape)
            start = 0
          #  print(data.shape[1])
            while start + window_size <= data.shape[1]:
                array = []
            #    print('data', data.shape)
                X = data[[channel],start: (start + window_size)]
                
             #   print('X[0]',X[0].shape)
              #  print('X',X.shape)
                array.append(np.array(X[0]))#仅添加了第一行，应该添加五个通道啊
                array.append(np.array(labels_4class))
             #   print(len(array))
                signal.append(np.array(array))
             #   print('signal',start,len(signal))#长度就是segments
                start = start + step_size
          #  print(signal.shape)
        signal = np.array(signal)
        data_list.append(signal)

        print('signal.shape',signal.shape,signal[:,1].shape,signal[0][0].shape)
        #                       (1160, 2)   (1160,)          (4, 512)  画出矩阵即可理解





        #np.save('F:\emotion\deap\data_pre\\eeg4\s'+q + sub, signal, allow_pickle=True, fix_imports=True)
pre_channel=[36,38,39]
#eeg_channel=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
eeg_channel=[0,1]#'Fp1', 'Fp2', 'F3', 'F4'from VMD+DNN
channel = []
channel.extend(eeg_channel)
channel.extend(pre_channel)
#以不重叠的1s划分时间段
window_size =512 # 512 #Averaging band power of 4 sec128
step_size =256 #256 #Each 2 sec update once128

data_list=[]
sum=0
for subjects in subject_list:
    epoching (subjects, channel, window_size, step_size)
    print(subjects)
    sum=sum+1
print(channel)
dl = np.stack(data_list, axis=0)#(n可取32, 1160, 2)
print(dl.shape)
np.save('/opt/data/private/DEAP/DATA/DATA_DL/mul5_4s2s'+q , dl, allow_pickle=True, fix_imports=True)



