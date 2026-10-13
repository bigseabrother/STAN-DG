import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from data_provider.m4 import M4Dataset, M4Meta
from data_provider.uea import subsample, interpolate_missing, Normalizer
from sktime.datasets import load_from_tsfile_to_dataframe
import warnings
import pickle
warnings.filterwarnings('ignore')

class Dataset_stock2geo(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path=['train.pkl','val.pkl','test.pkl'],
                 target='OT', scale=False, timeenc=0, freq='t', seasonal_patterns=None,use_time_features=False,
            stocknums=1027
                 ):
        # size [seq_len, label_len, pred_len]

        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.use_time_features = use_time_features
        
        self.root_path = root_path
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.data_path = data_path[type_map[flag]]
        # self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):


        with open(os.path.join(self.root_path,self.data_path), 'rb') as f:
            df_raw = pickle.load(f)
        # # 读取csv文件，根据train、val、test读取数据集
        # df_raw = pd.read_csv(os.path.join(self.root_path,
        #                                   self.data_path))
        print("读取数据集{},数据长度为{}".format( os.path.join(self.root_path,
                                          self.data_path),len(df_raw)))
        


        # self.data_x = df_raw[:,:,1:115] #除了time (1707, 471, 142) 115  280-9-4=267
        # self.data_y = df_raw[:,:,1:115] 
        selected_columns=[1, 2, 3, 4, 5, 6, 7, 8, 16, 18, 54, 59, 61, 72, 73, 78, 97, 114, 101, 102, 45, 46, 65]

        self.data_x = df_raw[:,:,selected_columns] #除了time (1707, 471, 142) 115
        self.data_y = df_raw[:,:,selected_columns] 
        print(self.data_x.shape)
        # self.data_x = np.concatenate((self.data_x,last_columns), axis=2)
        # self.data_y = np.concatenate((self.data_y,last_columns), axis=2)


        # self.data_year=df_raw[:,:,-1].reshape((df_raw.shape[0], df_raw.shape[1], 1))#(1707, 471, 1)

        # 
    #    'CYW', '流动比率(%)_CurRt', '速动比率(%)_QckRt', '现金比率(%)_CashRt',
    #    '权益乘数(%)_EquMul', '销售净利率(%)_NetPrfRt', '存货周转率(次)_InvtrTurovRt',
    #    '总资产周转率(次)_TotAssTurovRt', '总资产收益率（ROA）(%)_ROA',
    #    '每股收益(扣除)(元/股)_EPSCut'
        # self.data_stamp = df_raw[:,:,[117,119,120,121,122,124,125,126,127]] #4+4 9+4
        self.data_stamp = df_raw[:,:,-18:] #4+4 9+4

        # self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        # ( 23376,98, 96)
        seq_x = self.data_x[s_begin:s_end, :, :]
        seq_y = self.data_y[r_begin:r_end, :, :]#close
        # seq_year=self.data_year[s_begin:s_end,:,:] #year

        seq_x_mark = self.data_stamp[s_begin:s_end, :, :]
        seq_y_mark = self.data_stamp[r_begin:r_end, :, :]

        # return  _torch(seq_x, seq_y, None, None)
        return  _torch(seq_x, seq_y,seq_x_mark,seq_y_mark)


    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_stock3geo(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path=['train.pkl','val.pkl','test.pkl'],
                 target='OT', scale=False, timeenc=0, freq='t', seasonal_patterns=None,use_time_features=False,
                index_num=115,baseline=0,
            stocknums=1027
                 ):
        # size [seq_len, label_len, pred_len]

        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.index_num = index_num
        self.baseline = baseline
        self.stocknums=stocknums

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.use_time_features = use_time_features
        
        self.root_path = root_path
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.data_path = data_path[type_map[flag]]
        # self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):


        with open(os.path.join(self.root_path,self.data_path), 'rb') as f:
            df_raw = pickle.load(f)
        if  self.index_num==267: 
            with open(os.path.join('/disk4/cl/yc/yc/transformer/Time-Series-Library-main-transformer/data/dataset_financial2geo',self.data_path), 'rb') as f:
                geo = pickle.load(f)
        # # 读取csv文件，根据train、val、test读取数据集
        # df_raw = pd.read_csv(os.path.join(self.root_path,
        #                                   self.data_path))
        print("读取数据集{},数据长度为{}".format( os.path.join(self.root_path,
                                          self.data_path),len(df_raw)))
        


        # self.data_x = df_raw[:,:,1:115] #除了time (1707, 471, 142) 115  280-9-4=267
        # self.data_y = df_raw[:,:,1:115] 
        if  self.index_num==0:
            self.data_x = df_raw[:,:,1:9]#np.concatenate((df_raw[:,:,1:267],last_columns), axis=2)
            self.data_y = df_raw[:,:,1:9]#np.concatenate((df_raw[:,:,1:267],last_columns), axis=2)
            self.data_stamp = df_raw[:,:,-23:] #4+4 9+4 14+5+4

        elif  self.index_num==267: 
            self.data_x = df_raw[:,:,1:267]#np.concatenate((df_raw[:,:,1:267],last_columns), axis=2)
            self.data_y = df_raw[:,:,1:267]#np.concatenate((df_raw[:,:,1:267],last_columns), axis=2)
            self.data_stamp = geo[:,:,-23:] #4+4 9+4 14+5+4
        elif  self.index_num==14: 
            self.data_x = df_raw[:,:,1:115] #除了time (1707, 471, 142) 115  280-9-4=267
            self.data_y = df_raw[:,:,1:115] 
            self.data_stamp = df_raw[:,:,-23:-9] #4+4 9+4 14+5+4
        elif  self.index_num==5: 
            self.data_x = df_raw[:,:,1:115] #除了time (1707, 471, 142) 115  280-9-4=267
            self.data_y = df_raw[:,:,1:115] 
            self.data_stamp = df_raw[:,:,-9:-4] #4+4 9+4 14+5+4
        elif  self.index_num==4: 
            self.data_x = df_raw[:,:,1:115] #除了time (1707, 471, 142) 115  280-9-4=267
            self.data_y = df_raw[:,:,1:115] 
            self.data_stamp = df_raw[:,:,-4:] #4+4 9+4 14+5+4
        elif self.index_num==28:
            # 行业地理加在图增强里
            if self.stocknums==204:
                with open('/disk4/cl/yc/yc/transformer/Time-Series-Library-main-transformer/data/indusgeo300.pkl', 'rb') as f:
                    indusgeo = pickle.load(f)#.values
            elif self.stocknums==310:
                with open('/disk4/cl/yc/yc/transformer/Time-Series-Library-main-transformer/data/indusgeo500.pkl', 'rb') as f:
                    indusgeo = pickle.load(f)#.values
            elif self.stocknums==513:
                with open('/disk4/cl/yc/yc/transformer/Time-Series-Library-main-transformer/data/indusgeo1000.pkl', 'rb') as f:
                    indusgeo = pickle.load(f)#.values    # 将 indusgeo 数组扩展一个维度，以便沿着时间步维度拼接
            else:
                with open('/disk4/cl/yc/yc/transformer/Time-Series-Library-main-transformer/data/indusgeo.pkl', 'rb') as f:
                    indusgeo = pickle.load(f)#.values    # 将 indusgeo 数组扩展一个维度，以便沿着时间步维度拼接
            self.data_x = df_raw[:,:,1:115] #除了time (1707, 471, 142) 115  280-9-4=267
            self.data_y = df_raw[:,:,1:115] 
            # self.data_stamp = df_raw[:,:,-23:] #4+4 9+4 14+5+4
            indusgeo_expanded = np.tile(indusgeo.reshape(1, self.stocknums, 5), (self.data_x.shape[0], 1, 1))# np.tile(indusgeo[np.newaxis,:,  :], (self.data_x.shape[0],1,  1))
            # print(indusgeo_expanded.shape)
                # 沿着时间步维度拼接 indusgeo 到 data 上
            self.data_stamp = np.concatenate(( indusgeo_expanded,df_raw[:,:,-23:]), axis=2)
            assert self.data_stamp[0,0,0]==self.data_stamp[1,0,0],'拼接错误'
        else:
            self.data_x = df_raw[:,:,1:115] #除了time (1707, 471, 142) 115  280-9-4=267
            self.data_y = df_raw[:,:,1:115] 
            self.data_stamp = df_raw[:,:,-23:] #4+4 9+4 14+5+4


        if self.baseline!=0:
            # baseline 
            t,n,c=self.data_x.shape
            self.data_x = self.data_x.transpose(1, 0, 2).reshape(t*n, c) #除了time (1707, 471, 142) 115  280-9-4=267
            self.data_y =self.data_y.transpose(1, 0, 2).reshape(t*n, c)
            self.data_stamp = self.data_stamp.transpose(1, 0, 2).reshape(t*n, 23)

        print(self.data_x.shape)
        print(self.data_stamp.shape)


        # self.data_stamp = geo[:,:,-23:] #4+4 9+4 14+5+4

        # self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        # ( 23376,98, 96)
        seq_x = self.data_x[s_begin:s_end, :, :]
        seq_y = self.data_y[r_begin:r_end, :, :]#close
        # seq_year=self.data_year[s_begin:s_end,:,:] #year

        seq_x_mark = self.data_stamp[s_begin:s_end, :, :]
        seq_y_mark = self.data_stamp[r_begin:r_end, :, :]

        # return  _torch(seq_x, seq_y, None, None)
        return  _torch(seq_x, seq_y,seq_x_mark,seq_y_mark)


    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_stock(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path=['train.pkl','val.pkl','test.pkl'],
                 target='OT', scale=False, timeenc=0, freq='t', seasonal_patterns=None,use_time_features=False,
            stocknums=1027
                 ):
        # size [seq_len, label_len, pred_len]

        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.use_time_features = use_time_features
        
        self.root_path = root_path
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.data_path = data_path[type_map[flag]]
        # self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):


        with open(os.path.join(self.root_path,self.data_path), 'rb') as f:
            df_raw = pickle.load(f)
        # # 读取csv文件，根据train、val、test读取数据集
        # df_raw = pd.read_csv(os.path.join(self.root_path,
        #                                   self.data_path))
        print("读取数据集{},数据长度为{}".format( os.path.join(self.root_path,
                                          self.data_path),len(df_raw)))
        


        self.data_x = df_raw[:,:,1:115] #除了time (1707, 471, 142) 115  280-9-4=267
        self.data_y = df_raw[:,:,1:115] 
        # self.data_year=df_raw[:,:,-1].reshape((df_raw.shape[0], df_raw.shape[1], 1))#(1707, 471, 1)

        # 
    #    'CYW', '流动比率(%)_CurRt', '速动比率(%)_QckRt', '现金比率(%)_CashRt',
    #    '权益乘数(%)_EquMul', '销售净利率(%)_NetPrfRt', '存货周转率(次)_InvtrTurovRt',
    #    '总资产周转率(次)_TotAssTurovRt', '总资产收益率（ROA）(%)_ROA',
    #    '每股收益(扣除)(元/股)_EPSCut'
        # self.data_stamp = df_raw[:,:,[117,119,120,121,122,124,125,126,127]] #4+4 9+4
        self.data_stamp = df_raw[:,:,-13:] #4+4 9+4

        # self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        # ( 23376,98, 96)
        seq_x = self.data_x[s_begin:s_end, :, :]
        seq_y = self.data_y[r_begin:r_end, :, :]#close
        # seq_year=self.data_year[s_begin:s_end,:,:] #year

        seq_x_mark = self.data_stamp[s_begin:s_end, :, :]
        seq_y_mark = self.data_stamp[r_begin:r_end, :, :]

        # return  _torch(seq_x, seq_y, None, None)
        return  _torch(seq_x, seq_y,seq_x_mark,seq_y_mark)


    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
class Dataset_stock0index(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path=['train.pkl','val.pkl','test.pkl'],
                 target='OT', scale=False, timeenc=0, freq='t', seasonal_patterns=None,use_time_features=False,
                index_num=115,baseline=0,
            stocknums=1027

                 ):
        # size [seq_len, label_len, pred_len]

        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.index_num=index_num
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.use_time_features = use_time_features
        self.baseline=baseline
        self.root_path = root_path
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.data_path = data_path[type_map[flag]]
        # self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):


        with open(os.path.join(self.root_path,self.data_path), 'rb') as f:
            df_raw = pickle.load(f)
        # # 读取csv文件，根据train、val、test读取数据集
        # df_raw = pd.read_csv(os.path.join(self.root_path,
        #                                   self.data_path))
        print("读取数据集{},数据长度为{}".format( os.path.join(self.root_path,
                                          self.data_path),len(df_raw)))
        
        if  self.index_num==114: 
            # 没有基本面
            # last_columns= df_raw[:,:,-23:-4] #除了time (1707, 471, 142) 115

            self.data_x = df_raw[:,:,1:115]
            self.data_y = df_raw[:,:,1:115]
        elif  self.index_num==119: 
            # 没有财务
            last_columns= df_raw[:,:,-9:-4] #除了time (1707, 471, 142) 115
            self.data_x = np.concatenate((df_raw[:,:,1:115],last_columns), axis=2)
            self.data_y = np.concatenate((df_raw[:,:,1:115],last_columns), axis=2)
        elif  self.index_num==128: 
              # 没有地理
            last_columns= df_raw[:,:,-23:-9] #除了time (1707, 471, 142) 115
            self.data_x = np.concatenate((df_raw[:,:,1:115],last_columns), axis=2)
            self.data_y = np.concatenate((df_raw[:,:,1:115],last_columns), axis=2)      
        # self.data_year=df_raw[:,:,-1].reshape((df_raw.shape[0], df_raw.shape[1], 1))#(1707, 471, 1)
        else:
            last_columns= df_raw[:,:,-23:-4] #除了time (1707, 471, 142) 115

            self.data_x = np.concatenate((df_raw[:,:,1:115],last_columns), axis=2)
            self.data_y = np.concatenate((df_raw[:,:,1:115],last_columns), axis=2)


        # self.data_stamp = df_raw[:,:,[117,119,120,121,122,124,125,126,127]] #4+4 9+4
        self.data_stamp = df_raw[:,:,-4:] #4+4 9+4


        if self.baseline!=0:
            # baseline 
            t,n,c=self.data_x.shape
            
            self.data_x = self.data_x.transpose(1, 0, 2).reshape(t*n, c) #除了time (1707, 471, 142) 115  280-9-4=267
            self.data_y =self.data_y.transpose(1, 0, 2).reshape(t*n, c)
            self.data_stamp = self.data_stamp.transpose(1, 0, 2).reshape(t*n, 4)
        # self.data_stamp = data_stamp
        print(self.data_x.shape)

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        # ( 23376,98, 96)
        seq_x = self.data_x[s_begin:s_end, :, :]
        seq_y = self.data_y[r_begin:r_end, :, :]#close
        # seq_year=self.data_year[s_begin:s_end,:,:] #year

        seq_x_mark = self.data_stamp[s_begin:s_end, :, :]
        seq_y_mark = self.data_stamp[r_begin:r_end, :, :]

        # return  _torch(seq_x, seq_y, None, None)
        return  _torch(seq_x, seq_y,seq_x_mark,seq_y_mark)


    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
class Dataset_stock36index(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path=['train.pkl','val.pkl','test.pkl'],
                 target='OT', scale=False, timeenc=0, freq='t', seasonal_patterns=None,use_time_features=False,
            stocknums=1027
                 ):
        # size [seq_len, label_len, pred_len]

        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.use_time_features = use_time_features
        
        self.root_path = root_path
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.data_path = data_path[type_map[flag]]
        # self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):


        with open(os.path.join(self.root_path,self.data_path), 'rb') as f:
            df_raw = pickle.load(f)
        # # 读取csv文件，根据train、val、test读取数据集
        # df_raw = pd.read_csv(os.path.join(self.root_path,
        #                                   self.data_path))
        print("读取数据集{},数据长度为{}".format( os.path.join(self.root_path,
                                          self.data_path),len(df_raw)))
        

        selected_columns=[ 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 16, 18, 19, 22, 23, 24, 25, 26, 27, 29, 30, 32, 33, 34, 35, 36, 40, 41, 43, 44, 45, 46, 51, 52, 53, 54, 57, 59, 61, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 77, 78, 80, 82, 84, 85, 86, 87, 89, 92, 93, 94, 95, 96, 97, 99, 100, 101, 102, 103, 104, 107, 108, 112, 114]

        self.data_x = df_raw[:,:,selected_columns] #除了time (1707, 471, 142) 115
        self.data_y = df_raw[:,:,selected_columns] 
        # self.data_year=df_raw[:,:,-1].reshape((df_raw.shape[0], df_raw.shape[1], 1))#(1707, 471, 1)

        # 
    #    'CYW', '流动比率(%)_CurRt', '速动比率(%)_QckRt', '现金比率(%)_CashRt',
    #    '权益乘数(%)_EquMul', '销售净利率(%)_NetPrfRt', '存货周转率(次)_InvtrTurovRt',
    #    '总资产周转率(次)_TotAssTurovRt', '总资产收益率（ROA）(%)_ROA',
    #    '每股收益(扣除)(元/股)_EPSCut'
        # self.data_stamp = df_raw[:,:,[117,119,120,121,122,124,125,126,127]] #4+4 9+4
        self.data_stamp = df_raw[:,:,-18:] #4+4 9+4

        # self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        # ( 23376,98, 96)
        seq_x = self.data_x[s_begin:s_end, :, :]
        seq_y = self.data_y[r_begin:r_end, :, :]#close
        # seq_year=self.data_year[s_begin:s_end,:,:] #year

        seq_x_mark = self.data_stamp[s_begin:s_end, :, :]
        seq_y_mark = self.data_stamp[r_begin:r_end, :, :]

        # return  _torch(seq_x, seq_y, None, None)
        return  _torch(seq_x, seq_y,seq_x_mark,seq_y_mark)


    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_stock45index(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path=['train.pkl','val.pkl','test.pkl'],
                 target='OT', scale=False, timeenc=0, freq='t', seasonal_patterns=None,use_time_features=False,
            stocknums=1027
                 ):
        # size [seq_len, label_len, pred_len]

        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.use_time_features = use_time_features
        
        self.root_path = root_path
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.data_path = data_path[type_map[flag]]
        # self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):


        with open(os.path.join(self.root_path,self.data_path), 'rb') as f:
            df_raw = pickle.load(f)
        # # 读取csv文件，根据train、val、test读取数据集
        # df_raw = pd.read_csv(os.path.join(self.root_path,
        #                                   self.data_path))
        print("读取数据集{},数据长度为{}".format( os.path.join(self.root_path,
                                          self.data_path),len(df_raw)))
        

        selected_columns=[1, 2, 3, 4, 5, 6, 7, 8, 16, 18, 54, 59, 61, 72, 73, 78, 97, 114, 101, 102, 45, 46, 65]

        # [ 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 16, 18, 19, 22, 23, 24, 25, 26, 27, 29, 30, 32, 33, 34, 35, 36, 40, 41, 43, 44, 45, 46, 51, 52, 53, 54, 57, 59, 61, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 77, 78, 80, 82, 84, 85, 86, 87, 89, 92, 93, 94, 95, 96, 97, 99, 100, 101, 102, 103, 104, 107, 108, 112, 114]
        last_columns = df_raw[:, :,-18:-4]
        self.data_x = df_raw[:,:,selected_columns] #除了time (1707, 471, 142) 115
        self.data_y = df_raw[:,:,selected_columns] 
        # self.data_x =  np.hstack((self.data_x , last_columns))
        # self.data_y =  np.hstack((self.data_y , last_columns))
        self.data_x = np.concatenate((self.data_x,last_columns), axis=2)
        self.data_y = np.concatenate((self.data_y,last_columns), axis=2)
        print(self.data_x.shape)
        # self.data_year=df_raw[:,:,-1].reshape((df_raw.shape[0], df_raw.shape[1], 1))#(1707, 471, 1)

        # 
    #    'CYW', '流动比率(%)_CurRt', '速动比率(%)_QckRt', '现金比率(%)_CashRt',
    #    '权益乘数(%)_EquMul', '销售净利率(%)_NetPrfRt', '存货周转率(次)_InvtrTurovRt',
    #    '总资产周转率(次)_TotAssTurovRt', '总资产收益率（ROA）(%)_ROA',
    #    '每股收益(扣除)(元/股)_EPSCut'
        # self.data_stamp = df_raw[:,:,[117,119,120,121,122,124,125,126,127]] #4+4 9+4
        self.data_stamp = df_raw[:,:,-4:] #4+4 9+4

        # self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        # ( 23376,98, 96)
        seq_x = self.data_x[s_begin:s_end, :, :]
        seq_y = self.data_y[r_begin:r_end, :, :]#close
        # seq_year=self.data_year[s_begin:s_end,:,:] #year

        seq_x_mark = self.data_stamp[s_begin:s_end, :, :]
        seq_y_mark = self.data_stamp[r_begin:r_end, :, :]

        # return  _torch(seq_x, seq_y, None, None)
        return  _torch(seq_x, seq_y,seq_x_mark,seq_y_mark)


    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
class Dataset_custom_stock(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path=['train.pkl','val.pkl','test.pkl'],
                 target='OT', scale=False, timeenc=0, freq='t', seasonal_patterns=None,use_time_features=False,
            stocknums=1027
                 ):
        # size [seq_len, label_len, pred_len]

        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.use_time_features = use_time_features
        
        self.root_path = root_path
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.data_path = data_path[type_map[flag]]
        # self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):


        with open(os.path.join(self.root_path,self.data_path), 'rb') as f:
            df_raw = pickle.load(f)
        # # 读取csv文件，根据train、val、test读取数据集
        # df_raw = pd.read_csv(os.path.join(self.root_path,
        #                                   self.data_path))
        print("读取数据集{},数据长度为{}".format( os.path.join(self.root_path,
                                          self.data_path),len(df_raw)))
        


        last_columns= df_raw[:,:,-23:-4] #除了time (1707, 471, 142) 115

        self.data_x = np.concatenate((df_raw[:,:,:115],last_columns), axis=2)
        self.data_y = np.concatenate((df_raw[:,:,:115],last_columns), axis=2)

        print(self.data_x.shape)

        self.data_stamp = df_raw[:,:,-4:] 
        # self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        # ( 23376,98, 96)
        seq_x = self.data_x[s_begin:s_end, :, :]
        seq_y = self.data_y[r_begin:r_end, :, :]#close
        # seq_year=self.data_year[s_begin:s_end,:,:] #year

        seq_x_mark = self.data_stamp[s_begin:s_end, :, :]
        seq_y_mark = self.data_stamp[r_begin:r_end, :, :]

        return  _torch(seq_x, seq_y, seq_x_mark, seq_y_mark)


    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

def _torch(*dfs):
    # return tuple(torch.from_numpy(x.astype(np.float32) ).float() for x in dfs)
        # 将dfs解包为单独的变量
    # if len(dfs)==2:
    #     seq_x, seq_y = dfs#,seq_x_mark,seq_y_mark, seq_year = dfs
    # else:
    seq_x, seq_y ,seq_x_mark,seq_y_mark = dfs
    # seq_x, seq_y = dfs

    # 将seq_x和seq_y转换为np.float32类型
    seq_x_torch = torch.from_numpy(seq_x.astype(np.float32))
    seq_y_torch = torch.from_numpy(seq_y.astype(np.float32))
    seq_x_mark_torch = torch.from_numpy(seq_x_mark.astype(np.int64))
    seq_y_mark_torch = torch.from_numpy(seq_y_mark.astype(np.int64))
    # seq_x_mark_torch = torch.from_numpy(seq_x_mark)
    # seq_y_mark_torch = torch.from_numpy(seq_y_mark)
    return (seq_x_torch, seq_y_torch,seq_x_mark_torch,seq_y_mark_torch)
    # return (seq_x_torch, seq_y_torch, None, None)
    # if len(dfs)==5:
    #     # 将seq_year转换为int类型
    #     # seq_year_torch = torch.from_numpy(seq_year.astype(np.int64))

    #     # 将转换后的Tensors作为元组返回
    #     return (seq_x_torch, seq_y_torch)#,seq_x_mark_torch,seq_y_mark_torch ,seq_year_torch)
    # else:
    #     return (seq_x_torch, seq_y_torch)#,seq_x_mark_torch,seq_y_mark_torch)

