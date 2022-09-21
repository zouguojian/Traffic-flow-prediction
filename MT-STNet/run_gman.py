# -- coding: utf-8 --
'''
the shape of sparsetensor is a tuuple, like this
(array([[  0, 297],
       [  0, 296],
       [  0, 295],
       ...,
       [161,   2],
       [161,   1],
       [161,   0]], dtype=int32), array([0.00323625, 0.00485437, 0.00323625, ..., 0.00646204, 0.00161551,
       0.00161551], dtype=float32), (162, 300))
axis=0: is nonzero values, x-axis represents Row, y-axis represents Column.
axis=1: corresponding the nonzero value.
axis=2: represents the sparse matrix shape.
'''
from __future__ import division
from __future__ import print_function
from model.utils import *
from model.models import GCN
from model.hyparameter import parameter
from model.embedding import embedding
from model.sp_dis import sp_dis
from model.degree import in_out_deg
from baseline.gman.model import GMAN

import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import model.data_next as data_load
import os
import argparse
import csv
import datetime

tf.reset_default_graph()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
logs_path = "board"

# os.environ['CUDA_VISIBLE_DEVICES']='2'
#
# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession
#
# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)


class Model(object):
    def __init__(self, hp):
        '''
        :param para:
        '''
        self.hp = hp             # hyperparameter
        self.init_gcn()          # init gcn model
        self.init_placeholder()  # init placeholder
        self.init_embed()        # init embedding
        self.model()             # init prediction model

    def init_gcn(self):
        '''
        :return:
        '''
        self.adj = preprocess_adj(self.adjecent())

        # define gcn model
        if self.hp.model_name == 'gcn_cheby':
            self.support = chebyshev_polynomials(self.adj, self.hp.max_degree)
            self.num_supports = 1 + self.hp.max_degree
            self.model_func = GCN
        else:
            self.support = [self.adj]
            self.num_supports = 1
            self.model_func = GCN

    def init_placeholder(self):
        '''
        :return:
        '''
        self.placeholders = {
            'sp':tf.placeholder(tf.int32, shape=(self.hp.site_num*self.hp.site_num, 15), name='input_sp'),
            'dis':tf.placeholder(tf.int32, shape=(self.hp.site_num, self.hp.site_num), name='input_dis'),
            'in_deg': tf.placeholder(tf.int32, shape=(1,self.hp.site_num), name='input_in_deg'),
            'out_deg': tf.placeholder(tf.int32, shape=(1, self.hp.site_num), name='input_out_deg'),
            'position': tf.placeholder(tf.int32, shape=(1, self.hp.site_num), name='input_position'),
            'day': tf.placeholder(tf.int32, shape=(None, self.hp.site_num), name='input_day'),
            'hour': tf.placeholder(tf.int32, shape=(None, self.hp.site_num), name='input_hour'),
            'minute': tf.placeholder(tf.int32, shape=(None, self.hp.site_num), name='input_minute'),
            'indices_i': tf.placeholder(dtype=tf.int64, shape=[None, None], name='input_indices'),
            'values_i': tf.placeholder(dtype=tf.float32, shape=[None], name='input_values'),
            'dense_shape_i': tf.placeholder(dtype=tf.int64, shape=[None], name='input_dense_shape'),
            'features': tf.placeholder(tf.float32, shape=[None, self.hp.site_num, self.hp.features], name='input_features'),
            'labels': tf.placeholder(tf.float32, shape=[None, self.hp.site_num, self.hp.output_length], name='labels'),
            'dropout': tf.placeholder_with_default(0., shape=(), name='input_dropout'),
            'num_features_nonzero': tf.placeholder(tf.int32, name='input_zero')  # helper variable for sparse dropout
        }

        self.supports = [tf.SparseTensor(indices=self.placeholders['indices_i'],
                                         values=self.placeholders['values_i'],
                                         dense_shape=self.placeholders['dense_shape_i']) for _ in range(self.num_supports)]

    def adjecent(self):
        '''
        :return: adjacent matrix
        '''
        data = pd.read_csv(filepath_or_buffer=self.hp.file_adj)
        adj = np.zeros(shape=[self.hp.site_num, self.hp.site_num])
        for line in data[['src_FID', 'nbr_FID']].values:
            adj[line[0]][line[1]] = 1
        return adj

    def init_embed(self):
        '''
        :return:
        '''
        self.sp_emb = embedding(self.placeholders['sp'],vocab_size=self.hp.edge_num+1, num_units=self.hp.emb_size,scale=False, scope="sp_embed")
        self.dis_emb = embedding(self.placeholders['dis'],vocab_size=734, num_units=1, scale=False, scope="dis_embed")
        self.in_deg_emb = embedding(self.placeholders['in_deg'], vocab_size=5, num_units=self.hp.emb_size, scale=False, scope="in_embed")
        self.out_deg_emb = embedding(self.placeholders['out_deg'], vocab_size=5, num_units=self.hp.emb_size, scale=False, scope="out_embed")

        # self.edge_dis = tf.get_variable('dis_embed',
        #                                 dtype=tf.float32,
        #                                 shape=[self.hp.site_num*self.hp.site_num, 1],
        #                                 initializer=tf.truncated_normal_initializer(mean=0, stddev=1, seed=0))
        with tf.variable_scope('position'):
            p_emd = embedding(self.placeholders['position'], vocab_size=self.hp.site_num,num_units=self.hp.emb_size,scale=False, scope="position_embed")
            p_emd = tf.reshape(p_emd, shape=[1, self.hp.site_num, self.hp.emb_size])
            p_emd = tf.expand_dims(p_emd, axis=0)
            self.p_emd = tf.tile(p_emd, [self.hp.batch_size, self.hp.input_length+self.hp.output_length, 1, 1])
        with tf.variable_scope('day'):
            self.d_emb = embedding(self.placeholders['day'], vocab_size=32, num_units=self.hp.emb_size,scale=False, scope="day_embed")
            self.d_emd = tf.reshape(self.d_emb,
                                    shape=[self.hp.batch_size, self.hp.input_length + self.hp.output_length,
                                           self.hp.site_num, self.hp.emb_size])
        with tf.variable_scope('hour'):
            self.h_emb = embedding(self.placeholders['hour'], vocab_size=24, num_units=self.hp.emb_size,scale=False, scope="hour_embed")
            self.h_emd = tf.reshape(self.h_emb,
                                    shape=[self.hp.batch_size, self.hp.input_length + self.hp.output_length,
                                           self.hp.site_num, self.hp.emb_size])
        with tf.variable_scope('mimute'):
            self.m_emb = embedding(self.placeholders['minute'], vocab_size=12, num_units=self.hp.emb_size,scale=False, scope="minute_embed")
            self.m_emd = tf.reshape(self.m_emb,
                                    shape=[self.hp.batch_size, self.hp.input_length + self.hp.output_length,
                                           self.hp.site_num, self.hp.emb_size])
    def model(self):
        '''
        :return:
        '''
        print('#................................in the encoder step......................................#')
        features = tf.reshape(self.placeholders['features'], shape=[self.hp.batch_size,
                                                                    self.hp.input_length,
                                                                    self.hp.site_num])
        self.pre = GMAN(X=features,
                         TE=[self.h_emd, self.m_emd],
                         SE=self.p_emd,
                         P=self.hp.input_length,
                         Q=self.hp.output_length,
                         T=2,
                         L=self.hp.num_blocks,
                         K=self.hp.num_heads,
                         d=self.hp.emb_size // self.hp.num_heads,
                         bn=False,
                         bn_decay=0.99,
                         is_training=self.hp.is_training)

        print('pres shape is : ', self.pre.shape)

        self.loss = tf.reduce_mean(
                tf.sqrt(tf.reduce_mean(tf.square(self.pre + 1e-10 - self.placeholders['labels']), axis=0)))
        self.train_op = tf.train.AdamOptimizer(self.hp.learning_rate).minimize(self.loss)

    def test(self):
        '''
        :return:
        '''
        model_file = tf.train.latest_checkpoint('weights/')
        self.saver.restore(self.sess, model_file)

    def describe(self, label, predict):
        '''
        :param label:
        :param predict:
        :return:
        '''
        plt.figure()
        # Label is observed value,Blue
        plt.plot(label[0:], 'b', label=u'actual value')
        # Predict is predicted value，Red
        plt.plot(predict[0:], 'r', label=u'predicted value')
        # use the legend
        plt.legend()
        # plt.xlabel("time(hours)", fontsize=17)
        # plt.ylabel("pm$_{2.5}$ (ug/m$^3$)", fontsize=17)
        # plt.title("the prediction of pm$_{2.5}", fontsize=17)
        plt.show()

    def initialize_session(self):
        self.sess = tf.Session()
        self.saver = tf.train.Saver(var_list=tf.trainable_variables())

    def re_current(self, a, max, min):
        return [num * (max - min) + min for num in a]

    def run_epoch(self):
        '''
        :return:
        '''
        max_rmse = 100
        self.sess.run(tf.global_variables_initializer())

        iterate = data_load.DataClass(hp=self.hp)
        train_next = iterate.next_batch(batch_size=self.hp.batch_size, epoch=self.hp.epoch, is_training=True)

        for i in range(int((iterate.length // self.hp.site_num * iterate.divide_ratio - (
                iterate.input_length + iterate.output_length)) // iterate.step)
                       * self.hp.epoch // self.hp.batch_size):
            x, day, hour, minute, label = self.sess.run(train_next)
            features = np.reshape(x, [-1, self.hp.site_num, self.hp.features])
            day = np.reshape(day, [-1, self.hp.site_num])
            hour = np.reshape(hour, [-1, self.hp.site_num])
            minute = np.reshape(minute, [-1, self.hp.site_num])
            sp,dis = sp_dis(hp=self.hp)
            in_deg, out_deg = in_out_deg(self.hp)
            feed_dict = construct_feed_dict(features, self.adj, label, day, hour, minute, self.placeholders, site_num=self.hp.site_num,sp=sp,dis=dis,in_deg=in_deg,out_deg=out_deg)
            feed_dict.update({self.placeholders['dropout']: self.hp.dropout})
            loss_, _ = self.sess.run((self.loss, self.train_op), feed_dict=feed_dict)
            print("after %d steps,the training average loss value is : %.6f" % (i, loss_))

            # validate processing
            if i % 100 == 0:
                rmse_error = self.evaluate()

                if max_rmse > rmse_error:
                    print("the validate average rmse loss value is : %.6f" % (rmse_error))
                    max_rmse = rmse_error
                    self.saver.save(self.sess, save_path=self.hp.save_path + 'model.ckpt')

                    # if os.path.exists('model_pb'): shutil.rmtree('model_pb')
                    # builder = tf.saved_model.builder.SavedModelBuilder('model_pb')
                    # builder.add_meta_graph_and_variables(self.sess, ["mytag"])
                    # builder.save()

    def evaluate(self):
        '''
        :return:
        '''
        label_list = list()
        predict_list = list()

        label_list1,label_list2,label_list3 = list(),list(),list()
        predict_list1,predict_list2,predict_list3 = list(),list(),list()

        # with tf.Session() as sess:
        model_file = tf.train.latest_checkpoint(self.hp.save_path)
        if not self.hp.is_training:
            print('the model weights has been loaded:')
            self.saver.restore(self.sess, model_file)
            # self.saver.save(self.sess, save_path='gcn/model/' + 'model.ckpt')

        iterate_test = data_load.DataClass(hp=self.hp)
        test_next = iterate_test.next_batch(batch_size=self.hp.batch_size, epoch=1, is_training=False)
        max, min = iterate_test.max_dict['flow'], iterate_test.min_dict['flow']
        print(max, min)

        # file = open('results/gman.csv', 'w', encoding='utf-8')
        # writer = csv.writer(file)
        # writer.writerow(['station']+['day_'+str(i) for i in range(self.hp.predict_length)]+['hour_'+str(i) for i in range(self.hp.predict_length)]+
        #                 ['minute_' + str(i) for i in range(self.hp.predict_length)]+['label_'+str(i) for i in range(self.hp.predict_length)]+
        #                 ['predict_' + str(i) for i in range(self.hp.predict_length)])

        for i in range(int((iterate_test.length // self.hp.site_num
                            - iterate_test.length // self.hp.site_num * iterate_test.divide_ratio
                            - (iterate_test.input_length + iterate_test.output_length)) // iterate_test.output_length)
                       // self.hp.batch_size):
            # start_time = datetime.datetime.now()
            x, day, hour, minute, label = self.sess.run(test_next)
            features = np.reshape(x, [-1, self.hp.site_num, self.hp.features])
            day = np.reshape(day, [-1, self.hp.site_num])
            hour = np.reshape(hour, [-1, self.hp.site_num])
            minute = np.reshape(minute, [-1, self.hp.site_num])
            sp, dis = sp_dis(hp=self.hp)
            in_deg, out_deg = in_out_deg(self.hp)
            feed_dict = construct_feed_dict(features, self.adj, label, day, hour, minute, self.placeholders, site_num=self.hp.site_num,sp=sp,dis=dis,in_deg=in_deg,out_deg=out_deg)
            feed_dict.update({self.placeholders['dropout']: 0.0})

            pre = self.sess.run((self.pre), feed_dict=feed_dict)
            label_list.append(label[:,:,:self.hp.predict_length])
            predict_list.append(pre[:,:,:self.hp.predict_length])

            # for site in range(self.hp.site_num):
            #     writer.writerow([station[0][site]]+list(day[self.hp.input_length:,0])+
            #                      list(hour[self.hp.input_length:,0])+
            #                      list(minute[self.hp.input_length:,0]*5)+
            #                      list(np.round(self.re_current(label[0][site],max,min)))+
            #                      list(np.round(self.re_current(pre[0][site],max,min))))

            label_list1.append(label[:,:13,:self.hp.predict_length])
            label_list2.append(label[:, 13:26, :self.hp.predict_length])
            label_list3.append(label[:, 26:, :self.hp.predict_length])
            predict_list1.append(pre[:, :13, :self.hp.predict_length])
            predict_list2.append(pre[:, 13:26, :self.hp.predict_length])
            predict_list3.append(pre[:, 26:, :self.hp.predict_length])
            # end_time = datetime.datetime.now()
            # total_time = end_time - start_time
            # print("Total running times is : %f" % total_time.total_seconds())

        label_list = np.reshape(np.array(label_list, dtype=np.float32),
                                [-1, self.hp.site_num, self.hp.predict_length]).transpose([1, 0, 2])
        predict_list = np.reshape(np.array(predict_list, dtype=np.float32),
                                  [-1, self.hp.site_num, self.hp.predict_length]).transpose([1, 0, 2])

        label_list1 = np.reshape(np.array(label_list1, dtype=np.float32),
                                [-1, 13, self.hp.predict_length]).transpose([1, 0, 2])
        predict_list1 = np.reshape(np.array(predict_list1, dtype=np.float32),
                                  [-1, 13, self.hp.predict_length]).transpose([1, 0, 2])
        label_list2 = np.reshape(np.array(label_list2, dtype=np.float32),
                                [-1, 13, self.hp.predict_length]).transpose([1, 0, 2])
        predict_list2 = np.reshape(np.array(predict_list2, dtype=np.float32),
                                  [-1, 13, self.hp.predict_length]).transpose([1, 0, 2])
        label_list3 = np.reshape(np.array(label_list3, dtype=np.float32),
                                [-1, 40, self.hp.predict_length]).transpose([1, 0, 2])
        predict_list3 = np.reshape(np.array(predict_list3, dtype=np.float32),
                                  [-1, 40, self.hp.predict_length]).transpose([1, 0, 2])

        if self.hp.normalize:
            label_list = np.array(
                [self.re_current(np.reshape(site_label, [-1]), max, min) for site_label in label_list])
            predict_list = np.array(
                [self.re_current(np.reshape(site_label, [-1]), max, min) for site_label in predict_list])

            label_list1 = np.array(
                [self.re_current(np.reshape(site_label, [-1]), max, min) for site_label in label_list1])
            predict_list1 = np.array(
                [self.re_current(np.reshape(site_label, [-1]), max, min) for site_label in predict_list1])
            label_list2 = np.array(
                [self.re_current(np.reshape(site_label, [-1]), max, min) for site_label in label_list2])
            predict_list2 = np.array(
                [self.re_current(np.reshape(site_label, [-1]), max, min) for site_label in predict_list2])
            label_list3 = np.array(
                [self.re_current(np.reshape(site_label, [-1]), max, min) for site_label in label_list3])
            predict_list3 = np.array(
                [self.re_current(np.reshape(site_label, [-1]), max, min) for site_label in predict_list3])
        else:
            label_list = np.array([np.reshape(site_label, [-1]) for site_label in label_list])
            predict_list = np.array([np.reshape(site_label, [-1]) for site_label in predict_list])

        label_list = np.reshape(label_list, [-1])
        predict_list = np.reshape(predict_list, [-1])

        label_list1 = np.reshape(label_list1, [-1])
        predict_list1 = np.reshape(predict_list1, [-1])
        label_list2 = np.reshape(label_list2, [-1])
        predict_list2 = np.reshape(predict_list2, [-1])
        label_list3 = np.reshape(label_list3, [-1])
        predict_list3 = np.reshape(predict_list3, [-1])

        # average_error, rmse_error, cor, R2 = accuracy(label_list, predict_list)  # 产生预测指标
        print('1')
        metric(predict_list1, label_list1)
        print('2')
        metric(predict_list2, label_list2)
        print('3')
        metric(predict_list3, label_list3)
        print('4')
        mae, rmse, mape, cor, r2=metric(predict_list,label_list)
        # self.describe(label_list, predict_list)   #预测值可视化
        return mae


def main(argv=None):
    '''
    :param argv:
    :return:
    '''
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

    pre_model = Model(para)
    pre_model.initialize_session()

    if int(val) == 1:
        pre_model.run_epoch()
    else:
        pre_model.evaluate()

    print('#...................................finished............................................#')


if __name__ == '__main__':
    main()