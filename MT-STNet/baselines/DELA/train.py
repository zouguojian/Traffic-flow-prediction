# -- coding: utf-8 --
from __future__ import division
from __future__ import print_function
from dela import DelaClass
from utils import *
from hyparameter import parameter
from embedding import embedding
from data_load import *
import numpy as np
import argparse
import datetime
import csv
import math
tf.reset_default_graph()
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
logs_path = "board"

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

class Model(object):
    def __init__(self, para, min, max):
        self.para = para
        self.min = min
        self.max = max
        self.input_len = self.para.input_length
        self.output_len = self.para.output_length
        self.total_len = self.input_len + self.output_len
        self.features = self.para.features
        self.batch_size = self.para.batch_size
        self.epochs = self.para.epoch
        self.site_num = self.para.site_num
        self.emb_size = self.para.emb_size
        self.is_training = self.para.is_training
        self.learning_rate = self.para.learning_rate
        self.model_name = self.para.model_name
        self.granularity = self.para.granularity
        self.decay_epoch=self.para.decay_epoch
        self.num_train = 23967

        # placeholders
        self.placeholders = {
            'position': tf.placeholder(tf.int32, shape=(1, self.para.site_num), name='input_position'),
            'week': tf.placeholder(tf.int32, shape=(None, self.para.site_num), name='day_of_week'),
            'day': tf.placeholder(tf.int32, shape=(None, self.para.site_num), name='input_day'),
            'hour': tf.placeholder(tf.int32, shape=(None, self.para.site_num), name='input_hour'),
            'minute': tf.placeholder(tf.int32, shape=(None, self.para.site_num), name='input_minute'),
            'features': tf.placeholder(tf.float32, shape=[None, self.input_len, self.site_num, self.features],name='inputs'),
            'features_all': tf.placeholder(tf.float32, shape=[None, self.input_len+self.output_len, self.site_num, self.features],name='total_inputs'),
            'labels': tf.placeholder(tf.float32, shape=[None, self.site_num, self.total_len], name='labels'),
            'dropout': tf.placeholder_with_default(0., shape=(), name='input_dropout'),
            'random_mask': tf.placeholder(tf.int32, shape=(None, self.site_num, self.total_len), name='mask'),
            'is_training': tf.placeholder(shape=(), dtype=tf.bool)
        }
        self.embeddings()
        self.model()

    def embeddings(self):
        '''
        :return:
        '''
        p_emd = embedding(self.placeholders['position'], vocab_size=self.site_num, num_units=self.para.emb_size, scale=False, scope="position_embed")
        p_emd = tf.reshape(p_emd, shape=[1, self.site_num, self.emb_size])
        self.p_emd = tf.expand_dims(p_emd, axis=0)

        w_emb = embedding(self.placeholders['week'], vocab_size=7, num_units=self.emb_size, scale=False,
                          scope="day_of_week_embed")
        self.w_emd = tf.reshape(w_emb, shape=[-1, self.total_len, self.site_num, self.emb_size])

        d_emb = embedding(self.placeholders['day'], vocab_size=32, num_units=self.emb_size, scale=False, scope="day_embed")
        self.d_emd = tf.reshape(d_emb, shape=[-1, self.total_len, self.site_num, self.emb_size])

        h_emb = embedding(self.placeholders['hour'], vocab_size=24, num_units=self.para.emb_size, scale=False, scope="hour_embed")
        self.h_emd = tf.reshape(h_emb, shape=[-1, self.total_len, self.site_num, self.emb_size])

        m_emb = embedding(self.placeholders['minute'], vocab_size=24 * 60 //self.granularity, num_units=self.para.emb_size, scale=False, scope="minute_embed")
        self.m_emd = tf.reshape(m_emb, shape=[-1, self.total_len, self.site_num, self.emb_size])

    def model(self):
        '''
        :return:
        '''
        global_step = tf.Variable(0, trainable=False)

        features = tf.reshape(self.placeholders['features'], shape=[-1,
                                                                    self.input_len,
                                                                    self.site_num,
                                                                    self.features])
        in_hour = self.h_emd[:, :self.input_len]
        in_mimute = self.m_emd[:, :self.input_len]
        in_position = self.p_emd[:, :self.input_len]
        embeddings = [in_hour, in_mimute]
        # this step use to encoding the input series data
        encoder_init = DelaClass(self.para, placeholders=self.placeholders)
        x = encoder_init.encoding(features)
        # decoder
        pre = encoder_init.decoding(x, embeddings, index_emb=in_position)
        print('pres shape is : ', pre.shape)
        pre = tf.transpose(pre, [0, 2, 1])
        pre = pre * (self.max) + self.min
        self.pre = tf.transpose(pre, [0, 2, 1], name='output_y')
        print('prediction values shape is : ', self.pre.shape)

        learning_rate = tf.train.exponential_decay(
            self.learning_rate, global_step,
            decay_steps=self.decay_epoch * self.num_train // self.batch_size,
            decay_rate=0.7, staircase=True)
        learning_rate = tf.maximum(learning_rate, 1e-5)
        self.loss = mae_los(self.pre, self.placeholders['labels'][:,:,self.input_len:])
        self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss, global_step=global_step)

    def test(self):
        '''
        :return:
        '''
        model_file = tf.train.latest_checkpoint('weights/')
        self.saver.restore(self.sess, model_file)

    def initialize_session(self,session):
        self.sess = session
        self.saver = tf.train.Saver()

    def run_epoch(self,trainX, trainDoW, trainD, trainH, trainM, trainL, trainXAll, valX, valDoW, valD, valH, valM, valL, valXAll, testX, testDoW, testD, testH, testM, testL, testXAll):
        max_mae = 100
        shape = trainX.shape
        num_batch = math.ceil(shape[0] / self.batch_size)
        self.num_train=shape[0]
        self.sess.run(tf.global_variables_initializer())
        start_time = datetime.datetime.now()
        iteration=0
        for epoch in range(self.epochs):
            # shuffle
            permutation = np.random.permutation(shape[0])
            trainX = trainX[permutation]
            trainDoW = trainDoW[permutation]
            trainD = trainD[permutation]
            trainH = trainH[permutation]
            trainM = trainM[permutation]
            trainL = trainL[permutation]
            trainXAll = trainXAll[permutation]
            for batch_idx in range(num_batch):
                iteration+=1
                start_idx = batch_idx * self.batch_size
                end_idx = min(shape[0], (batch_idx + 1) * self.batch_size)
                random_mask = np.random.randint(low=0,high=2,size=[self.batch_size, self.site_num, self.total_len],dtype=np.int)
                xs = np.expand_dims(trainX[start_idx : end_idx], axis=-1)
                d_of_week = np.reshape(trainDoW[start_idx : end_idx], [-1, self.site_num])
                day = np.reshape(trainD[start_idx : end_idx], [-1, self.site_num])
                hour = np.reshape(trainH[start_idx : end_idx], [-1, self.site_num])
                minute = np.reshape(trainM[start_idx : end_idx], [-1, self.site_num])
                labels = trainL[start_idx : end_idx]
                xs_all = np.expand_dims(trainXAll[start_idx : end_idx], axis=-1)
                feed_dict = construct_feed_dict(xs, xs_all, labels, d_of_week, day, hour, minute, mask=random_mask, placeholders=self.placeholders, site=self.site_num)
                feed_dict.update({self.placeholders['dropout']: self.para.dropout})
                l,_ = self.sess.run((self.loss, self.train_op), feed_dict=feed_dict)
                # print("after %d steps and %d epochs, the training total average loss is : %.6f" % (batch_idx, epoch+1, l))

                if iteration == 100:
                    end_time = datetime.datetime.now()
                    total_time = end_time - start_time
                    print("Total running times is : %f" % total_time.total_seconds())
            print('validation')
            mae = self.evaluate(valX, valDoW, valD, valH, valM, valL, valXAll) # validate processing
            if max_mae > mae:
                print("in the %dth epoch, the validate average loss value is : %.3f" % (epoch+1, mae))
                max_mae = mae
                self.saver.save(self.sess, save_path=self.para.save_path)
            # print('testing')
            # self.evaluate(testX, testDoW, testD, testH, testM, testL, testXAll)

    def evaluate(self, testX, testDoW, testD, testH, testM, testL, testXAll):
        '''
        :return:
        '''
        labels_list, pres_list = list(), list()

        if not self.is_training:
            # model_file = tf.train.latest_checkpoint(self.para.save_path)
            saver = tf.train.import_meta_graph(self.para.save_path + '.meta')
            # saver.restore(sess, args.model_file)
            print('the model weights has been loaded:')
            saver.restore(self.sess, self.para.save_path)

        parameters = 0
        for variable in tf.trainable_variables():
            parameters += np.product([x.value for x in variable.get_shape()])
        print('trainable parameters: {:,}'.format(parameters))

        textX_shape = testX.shape
        total_batch = math.ceil(textX_shape[0] / self.batch_size)
        start_time = datetime.datetime.now()
        for b_idx in range(total_batch):
            start_idx = b_idx * self.batch_size
            end_idx = min(textX_shape[0], (b_idx + 1) * self.batch_size)
            random_mask = np.ones(shape=[self.batch_size,self.site_num,self.total_len],dtype=np.int)
            xs = np.expand_dims(testX[start_idx: end_idx], axis=-1)
            d_of_week = np.reshape(testDoW[start_idx: end_idx], [-1, self.site_num])
            day = np.reshape(testD[start_idx: end_idx], [-1, self.site_num])
            hour = np.reshape(testH[start_idx: end_idx], [-1, self.site_num])
            minute = np.reshape(testM[start_idx: end_idx], [-1, self.site_num])
            labels = testL[start_idx: end_idx]
            xs_all = np.expand_dims(testXAll[start_idx: end_idx], axis=-1)
            feed_dict = construct_feed_dict(xs, xs_all, labels, d_of_week, day, hour, minute, mask=random_mask, placeholders=self.placeholders,site=self.site_num, is_training=False)
            feed_dict.update({self.placeholders['dropout']: 0.0})
            pre_s = self.sess.run((self.pre), feed_dict=feed_dict)

            labels_list.append(labels[:, :, self.input_len:])
            pres_list.append(pre_s)

        end_time = datetime.datetime.now()
        total_time = end_time - start_time
        print("Total running times is : %f" % total_time.total_seconds())

        labels_list = np.concatenate(labels_list, axis=0)
        pres_list = np.concatenate(pres_list, axis=0)
        np.savez_compressed('data/DELA-' + 'YINCHUAN', **{'prediction': pres_list, 'truth': labels_list})

        print('                MAE\t\tRMSE\t\tMAPE')
        for (l,r) in [(0,13),(13,26),(26,66)]:
            for i in range(self.para.output_length):
                mae, rmse, mape = metric(pres_list[:,l:r,i], labels_list[:,l:r,i])
                print('step: %02d         %.3f\t\t%.3f\t\t%.3f%%' % (i + 1, mae, rmse, mape * 100))
            mae, rmse, mape = metric(pres_list[:,l:r], labels_list[:,l:r])  # 产生预测指标
            print('average:         %.3f\t\t%.3f\t\t%.3f%%' %(mae, rmse, mape * 100))
            print('\n')

        # describe(label_list, predict_list)   #预测值可视化
        # mae, rmse, mape = metric(pres_list, labels_list)  # 产生预测指标
        return mae


def main(argv=None):
    '''
    :param argv:
    :return:
    '''
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    print('#......................................beginning........................................#')
    para = parameter(argparse.ArgumentParser())
    para = para.get_para()

    print('Please input a number : 1 or 0. (1 and 0 represents the training or testing, respectively).')
    val = input('please input the number : ')

    if int(val) == 1:
        para.is_training = True
    else:
        para.batch_size = 1
        para.is_training = False

    trainX, trainDoW, trainD, trainH, trainM, trainL, trainXAll, valX, valDoW, valD, valH, valM, valL, valXAll, testX, testDoW, testD, testH, testM, testL, testXAll, min, max = loadData(para)
    print('trainX: %s\ttrainY: %s' % (trainX.shape, trainL.shape))
    print('valX:   %s\t\tvalY:   %s' % (valX.shape, valL.shape))
    print('testX:  %s\t\ttestY:  %s' % (testX.shape, testL.shape))
    print('data loaded!')
    pre_model = Model(para, min, max)
    pre_model.initialize_session(session)
    if int(val) == 1:
        pre_model.run_epoch(trainX, trainDoW, trainD, trainH, trainM, trainL, trainXAll, valX, valDoW, valD, valH, valM, valL, valXAll, testX, testDoW, testD, testH, testM, testL, testXAll)
    else:
        pre_model.evaluate(testX, testDoW, testD, testH, testM, testL, testXAll)

    print('#...................................finished............................................#')


if __name__ == '__main__':
    main()