# -- coding: utf-8 --

import tensorflow as tf
import numpy as np
import argparse
from model.hyparameter import parameter
import os
import csv
import pandas as pd

train_path=r'data/training_data/train.csv'

class DataClass(object):             # 切记这里的训练时段和测试时段的所用的对象不变，否则需要重复加载数据
    def __init__(self, hp=None):
        '''
        :param hp:
        '''
        self.hp = hp                              # hyperparameter
        self.min_value=0.000000000001
        self.input_length=hp.input_length         # time series length of input
        self.output_length=hp.output_length       # the length of prediction
        self.is_training=hp.is_training           # true or false
        self.data_divide=hp.divide_ratio          # the divide between in training set and test set ratio
        self.step=hp.step                         # windows step
        self.site_num=hp.site_num
        self.normalize = hp.normalize             # data normalization

        self.get_data(hp.file_train)
        self.length=self.data.shape[0]            # data length
        self.get_max_min(self.data)               # max and min values' list

        if self.normalize:
            self.normalization(data=self.data,index=6,max_list=self.max_list,min_list=self.min_list) # normalization

    def get_data(self,file_path=None):
        '''
        :param file_path:
        :return:
        '''
        self.data = pd.read_csv(file_path, encoding='utf-8')

    def get_max_min(self,data=None):
        '''
        :return: the max and min value of input features
        '''
        self.min_list=[]
        self.max_list=[]
        # print('the shape of features is :',self.data.values.shape[1])
        for i in range(data.values.shape[1]):
            self.min_list.append(min(data[list(data.keys())[i]].values))
            self.max_list.append(max(data[list(data.keys())[i]].values))
        print('the max feature list is :', self.max_list)
        print('the min feature list is :', self.min_list)

    def normalization(self,data=None,index=1,max_list=[],min_list=[]):
        keys=list(data.keys())

        for i in range(index,len(keys)):
            data[keys[i]]=(data[keys[i]] - np.array(min_list[i])) / (np.array(max_list[i]) - np.array(min_list[i]+self.min_value))

    def generator_(self):
        '''
        :return: yield the data of every time,
        shape:input_series:[time_size,field_size]
        label:[predict_size]
        '''

        if self.is_training:
            low,high=24*6*self.site_num, int(self.data.shape[0]//self.site_num * self.data_divide)*self.site_num
        else:
            low,high=int(self.data.shape[0]//self.site_num * self.data_divide) *self.site_num, self.data.shape[0]

        while low+self.site_num*(self.input_length + self.output_length)<= high:
            label=self.data.values[low + self.input_length * self.site_num: low + (self.input_length + self.output_length) * self.site_num,-2:-1]
            label=np.concatenate([label[i * self.site_num:(i + 1) * self.site_num, :] for i in range(self.output_length)], axis=1)

            yield (self.data.values[low:low+self.input_length*self.site_num, 6:7],
                   self.data.values[low:low+(self.input_length+self.output_length)*self.site_num, 4],
                   self.data.values[low:low + (self.input_length+self.output_length)* self.site_num, 5],
                   label)
            if self.is_training:
                low += self.step * self.site_num
            else:
                low+=self.output_length * self.site_num
        return

    def next_batch(self,batch_size,epochs, is_training=True):
        '''
        :return the iterator!!!
        :param batch_size:
        :param epochs:
        :return:
        '''
        self.is_training=is_training
        dataset=tf.data.Dataset.from_generator(self.generator_,output_types=(tf.float32,tf.int32, tf.int32, tf.float32))

        if self.is_training:
            dataset=dataset.shuffle(buffer_size=int(self.data.values.shape[0]//self.hp.site_num * self.data_divide-self.input_length-self.out_length)//self.step)
            dataset=dataset.repeat(count=epochs)
        dataset=dataset.batch(batch_size=batch_size)
        iterator=dataset.make_one_shot_iterator()

        return iterator.get_next()
# #
if __name__=='__main__':
    para = parameter(argparse.ArgumentParser())
    para = para.get_para()

    iter=DataClass(hp=para)
    print(iter.data.keys())
    # print(iter.data.loc[iter.data['ZoneID']==0])
    next=iter.next_batch(1,1, False)
    with tf.Session() as sess:
        for _ in range(4):
            x,y=sess.run(next)
            print(x.shape)
            print(y.shape)
            print(x[0])
            print(y[0])