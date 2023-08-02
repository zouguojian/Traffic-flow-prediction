# -- coding: utf-8 --

from __future__ import division
from __future__ import print_function

from tgcn import tgcnCell
from utils import *
from hyparameter import parameter
import matplotlib.pyplot as plt
from data_load import *
from inits import *

tf.reset_default_graph()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
logs_path = "board"

os.environ['CUDA_VISIBLE_DEVICES']='5'
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
#
# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)


class Model(object):
    def __init__(self, para, mean, std):
        '''
        :param para:
        '''
        self.para = para
        self.mean = mean
        self.std = std
        self.input_len = self.para.input_length
        self.output_len = self.para.output_length
        self.total_len = self.input_len + self.output_len
        self.features = self.para.features
        self.batch_size = self.para.batch_size
        self.epochs = self.para.epoch
        self.site_num = self.para.site_num
        self.emb_size = self.para.emb_size
        self.hidden_size = self.para.hidden_size
        self.is_training = self.para.is_training
        self.learning_rate = self.para.learning_rate
        self.model_name = self.para.model_name
        self.granularity = self.para.granularity
        self.num_train = 23967

        self.init_placeholder()  # init placeholder
        self.model()             # init prediction model


    def init_placeholder(self):
        '''
        :return:
        '''
        self.placeholders = {
            'features': tf.placeholder(tf.float32, shape=[None, self.input_len, self.site_num], name='input_features'),
            'labels': tf.placeholder(tf.float32, shape=[None, self.site_num, self.total_len], name='labels'),
            'dropout': tf.placeholder_with_default(0., shape=(), name='input_dropout')
        }

    def adjecent(self):
        '''
        :return: adjacent matrix
        '''
        data = pd.read_csv(filepath_or_buffer=self.para.file_adj)
        adj = np.zeros(shape=[self.site_num, self.site_num])
        for line in data[['src_FID', 'nbr_FID']].values:
            adj[line[0]][line[1]] = 1
        return adj

    def model(self):
        '''
        :return:
        '''

        def TGCN(_X, adj):
            ###
            cell_1 = tgcnCell(num_units=self.hidden_size, adj=adj, num_nodes=self.site_num)
            cell = tf.nn.rnn_cell.MultiRNNCell([cell_1], state_is_tuple=True)  # 可用多层
            _X = tf.unstack(_X, axis=1)
            outputs, states = tf.nn.static_rnn(cell, _X, dtype=tf.float32)
            print('outputs length is : ', len(outputs))
            print('outputs shape is : ', outputs[-1].shape)
            m = []
            for i in outputs:
                o = tf.reshape(i, shape=[-1, self.site_num, self.hidden_size])
                o = tf.reshape(o, shape=[-1, self.hidden_size])
                m.append(o)
            last_output = m[-1]
            print('last_output shape is : ', last_output.shape)
            last_output = tf.reshape(last_output, [-1, self.site_num, self.hidden_size])
            last_output = tf.reshape(last_output, [-1, self.site_num, self.hidden_size])
            last_output = tf.layers.dense(inputs=last_output, units=64, activation=tf.nn.relu, name='layer_1')
            output = tf.layers.dense(inputs=last_output, units=self.output_len, name='output_y')
            return output, m, states

        adj = self.adjecent()

        self.pre, _, _ = TGCN(self.placeholders['features'],adj=adj)
        self.pre = self.pre * (self.std) + self.mean
        print('pres shape is : ', self.pre.shape)

        self.loss = mae_los(self.pre, self.placeholders['labels'][:,:,self.input_len:])
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def test(self):
        '''
        :return:
        '''
        model_file = tf.train.latest_checkpoint('weights/')
        self.saver.restore(self.sess, model_file)

    def initialize_session(self,session):
        self.sess = session
        self.saver = tf.train.Saver()

    def run_epoch(self, trainX, trainL, valX, valL):
        '''
        from now on,the model begin to training, until the epoch to 100
        '''
        max_mae = 100
        shape = trainX.shape
        num_batch = math.ceil(shape[0] / self.batch_size)
        self.num_train=shape[0]
        self.sess.run(tf.global_variables_initializer())
        start_time = datetime.datetime.now()
        iteration=1
        for epoch in range(self.epochs):
            # shuffle
            permutation = np.random.permutation(shape[0])
            trainX = trainX[permutation]
            trainL = trainL[permutation]
            for batch_idx in range(num_batch):
                iteration+=1
                start_idx = batch_idx * self.batch_size
                end_idx = min(shape[0], (batch_idx + 1) * self.batch_size)
                xs = trainX[start_idx : end_idx]
                labels = trainL[start_idx : end_idx]
                feed_dict = construct_feed_dict(features=xs,
                                                labels=labels,
                                                placeholders=self.placeholders)
                feed_dict.update({self.placeholders['dropout']: self.para.dropout})
                loss_, _ = self.sess.run((self.loss, self.train_op), feed_dict=feed_dict)

                if iteration == 100:
                    end_time = datetime.datetime.now()
                    total_time = end_time - start_time
                    print("Total running times is : %f" % total_time.total_seconds())

            print('validation')
            mae = self.evaluate(valX, valL)  # validate processing
            if max_mae > mae:
                print("in the %dth epoch, the validate average loss value is : %.3f" % (epoch + 1, mae))
                max_mae = mae
                self.saver.save(self.sess, save_path=self.para.save_path)

    def evaluate(self, testX, testL):
        '''
        :param para:
        :param pre_model:
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
            xs = testX[start_idx: end_idx]
            labels = testL[start_idx: end_idx]
            feed_dict = construct_feed_dict(features=xs,
                                            labels=labels,
                                            placeholders=self.placeholders)
            feed_dict.update({self.placeholders['dropout']: 0.0})
            pre = self.sess.run((self.pre), feed_dict=feed_dict)

            labels_list.append(labels[:,:,self.input_len:])
            pres_list.append(pre)

        end_time = datetime.datetime.now()
        total_time = end_time - start_time
        print("Total running times is : %f" % total_time.total_seconds())

        labels_list = np.concatenate(labels_list, axis=0)
        pres_list = np.concatenate(pres_list, axis=0)
        np.savez_compressed('data/T-GCN-' + 'YINCHUAN', **{'prediction': pres_list, 'truth': labels_list})

        print('                MAE\t\tRMSE\t\tMAPE')
        for (l,r) in [(0,13),(13,26),(26,66)]:
            for i in range(self.para.output_length):
                mae, rmse, mape = metric(pres_list[:,l:r,i], labels_list[:,l:r,i])
                print('step: %02d         %.3f\t\t%.3f\t\t%.3f%%' % (i + 1, mae, rmse, mape * 100))
            mae, rmse, mape = metric(pres_list[:,l:r], labels_list[:,l:r])  # 产生预测指标
            print('average:         %.3f\t\t%.3f\t\t%.3f%%' %(mae, rmse, mape * 100))
            print('\n')
        return mae

def main(argv=None):
    '''
    :param argv:
    :return:
    '''

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.3
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

    trainX, trainDoW, trainM, trainL, trainXAll, valX, valDoW, valM, valL, valXAll, testX, testDoW, testM, testL, testXAll, mean, std = loadData(para)
    print('trainX: %s\ttrainY: %s' % (trainX.shape, trainL.shape))
    print('valX:   %s\t\tvalY:   %s' % (valX.shape, valL.shape))
    print('testX:  %s\t\ttestY:  %s' % (testX.shape, testL.shape))
    print('data loaded!')

    pre_model = Model(para, mean, std)
    pre_model.initialize_session(session)
    if int(val) == 1:
        pre_model.run_epoch(trainX, trainL, valX, valL)
    else:
        pre_model.evaluate(testX, testL)

    print('#...................................finished............................................#')


if __name__ == '__main__':
    main()