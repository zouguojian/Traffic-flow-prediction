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
from model.hyparameter import parameter
from model.embedding import embedding
from model.inits import *
from model.data_load import *
from model.st_block import STAttBlock, BridgeTrans, MT_STNet

tf.reset_default_graph()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
logs_path = "board"
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
tf.set_random_seed(1)


class Model(object):
    def __init__(self, hp, mean=0.0, std=1.0):
        '''
        :param para:
        '''
        self.hp = hp
        self.model_name = self.hp.model_name
        self.batch_size = self.hp.batch_size
        self.epochs = self.hp.epochs
        self.is_training = self.hp.is_training
        self.site_num = self.hp.site_num
        self.edge_num = self.hp.edge_num
        self.features = self.hp.features
        self.emb_size = self.hp.emb_size
        self.input_len = self.hp.input_length
        self.output_len = self.hp.output_length
        self.total_len = self.input_len + self.output_len
        self.num_blocks = self.hp.num_blocks
        self.granularity = self.hp.granularity
        self.scale = False
        self.num_supports = 1 + self.hp.max_degree if self.model_name == 'gcn_cheby' else 1
        self.decay_epoch=self.hp.decay_epoch
        self.num_train = 23967
        self.heads = self.hp.num_heads
        self.learning_rate = self.hp.learning_rate
        self.pre_len = self.hp.pre_len
        self.mean=mean
        self.std=std

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
            # self.num_supports = 1 + self.hp.max_degree
            # self.model_func = GCN
        else:
            self.support = [self.adj]
            # self.num_supports = 1
            # self.model_func = GCN

    def init_placeholder(self):
        '''
        :return:
        '''
        self.placeholders = {
            'sp':tf.placeholder(tf.int32, shape=(self.site_num*self.site_num, 15), name='input_sp'),
            'dis':tf.placeholder(tf.float32, shape=(self.site_num, self.site_num), name='input_dis'),
            'in_deg': tf.placeholder(tf.int32, shape=(1,self.site_num), name='input_in_deg'),
            'out_deg': tf.placeholder(tf.int32, shape=(1, self.site_num), name='input_out_deg'),
            'position': tf.placeholder(tf.int32, shape=(1, self.site_num), name='input_position'),
            'day_of_week': tf.placeholder(tf.int32, shape=(None, self.site_num), name='day_of_week'),
            'minute_of_day': tf.placeholder(tf.int32, shape=(None, self.site_num), name='minute_of_day'),
            'indices_i': tf.placeholder(dtype=tf.int64, shape=[None, None], name='input_indices'),
            'values_i': tf.placeholder(dtype=tf.float32, shape=[None], name='input_values'),
            'dense_shape_i': tf.placeholder(dtype=tf.int64, shape=[None], name='input_dense_shape'),
            'features': tf.placeholder(tf.float32, shape=[None, self.input_len, self.site_num, self.features], name='input_features'),
            'x_all': tf.placeholder(tf.float32, shape=[None, self.total_len, self.site_num, self.features], name='last_week'),
            'labels': tf.placeholder(tf.float32, shape=[None, self.site_num, self.output_len], name='labels'),
            'dropout': tf.placeholder_with_default(0., shape=(), name='input_dropout'),
            'is_training': tf.placeholder(shape=(), dtype=tf.bool),
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
        self.sp_em = embedding(self.placeholders['sp'], vocab_size=self.edge_num + 1, num_units=self.emb_size, scale=self.scale, scope="sp_emb")
        self.in_deg_em = embedding(self.placeholders['in_deg'], vocab_size=5, num_units=self.emb_size, scale=self.scale, scope="in_emb")
        self.out_deg_em = embedding(self.placeholders['out_deg'], vocab_size=5, num_units=self.emb_size, scale=self.scale, scope="out_emb")

        p_em = embedding(self.placeholders['position'], vocab_size=self.site_num, num_units=self.emb_size, scale=self.scale, scope="p_emb")
        self.p_em = tf.expand_dims(p_em, axis=0)

        d_o_wem = embedding(self.placeholders['day_of_week'], vocab_size=7, num_units=self.emb_size,scale=self.scale, scope="d_emb")
        self.d_o_wem = tf.reshape(d_o_wem, shape=[-1, self.total_len, self.site_num, self.emb_size])

        m_o_dem = embedding(self.placeholders['minute_of_day'], vocab_size=24 * 60 //self.granularity, num_units=self.emb_size, scale=self.scale, scope="m_emb")
        self.m_o_dem = tf.reshape(m_o_dem, shape=[-1, self.total_len, self.site_num, self.emb_size])

    def model(self):
        '''
        :return:
        '''
        global_step = tf.Variable(0, trainable=False)
        bn_momentum = tf.train.exponential_decay(0.5, global_step, decay_steps=self.decay_epoch * self.num_train // self.batch_size,
            decay_rate=0.5, staircase=True)
        bn_decay = tf.minimum(0.99, 1 - bn_momentum)

        pre = MT_STNet(X=self.placeholders['features'], 
                        X_all = self.placeholders['x_all'],
                        TE=[self.d_o_wem, self.m_o_dem], 
                        SE=self.p_em, 
                        P=self.input_len, 
                        Q=self.output_len, 
                        S=self.pre_len, 
                        L=self.num_blocks, 
                        K=self.heads, 
                        d=self.emb_size // self.heads, 
                        bn=True, 
                        bn_decay=bn_decay, 
                        is_training=self.placeholders['is_training'], 
                        supports=self.supports, 
                        placeholders=self.placeholders, 
                        spatial_inf=[self.placeholders['dis'], self.sp_em, self.in_deg_em, self.out_deg_em], 
                        hp=self.hp, 
                        model_name=self.model_name)

        pre = pre * self.std + self.mean
        self.pre = tf.transpose(pre, [0, 2, 1], name='output_y')
        print('predicted values\' shape is ', self.pre.shape)

        learning_rate = tf.train.exponential_decay(
            self.learning_rate, global_step,
            decay_steps=self.decay_epoch * self.num_train // self.batch_size,
            decay_rate=0.7, staircase=True)
        learning_rate = tf.maximum(learning_rate, 1e-5)

        self.loss = mae_los(self.pre, self.placeholders['labels'])
        self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss, global_step=global_step)

    def initialize_session(self,session):
        self.sess = session
        self.saver = tf.train.Saver()

    def run_epoch(self,trainX, trainDoW, trainD, trainH, trainM, trainL, trainXAll, valX, valDoW, valD, valH, valM, valL, valXAll, testX, testDoW, testD, testH, testM, testL, testXAll):
        '''
        :return:
        '''
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
            trainM = trainM[permutation]
            trainL = trainL[permutation]
            trainXAll = trainXAll[permutation]
            for batch_idx in range(num_batch):
                iteration+=1
                start_idx = batch_idx * self.batch_size
                end_idx = min(shape[0], (batch_idx + 1) * self.batch_size)
                xs = np.expand_dims(trainX[start_idx : end_idx], axis=-1)
                day_of_week = np.reshape(trainDoW[start_idx : end_idx], [-1, self.site_num])
                minute_of_day = np.reshape(trainM[start_idx : end_idx], [-1, self.site_num])
                labels = trainL[start_idx : end_idx]
                xs_all = np.expand_dims(trainXAll[start_idx : end_idx], axis=-1)
                sp, dis = sp_dis(hp=self.hp)
                in_deg, out_deg = in_out_deg(self.hp)
                feed_dict = construct_feed_dict(xs, xs_all, self.adj, labels, day_of_week, minute_of_day, self.placeholders, site_num=self.site_num, sp=sp, dis=dis, in_deg=in_deg, out_deg=out_deg)
                feed_dict.update({self.placeholders['dropout']: self.hp.dropout})
                l, _ = self.sess.run((self.loss, self.train_op), feed_dict=feed_dict)

                if iteration == 100:
                    end_time = datetime.datetime.now()
                    total_time = end_time - start_time
                    print("Total running times is : %f" % total_time.total_seconds())
            print('validation')
            mae = self.evaluate(valX, valDoW, valD, valH, valM, valL, valXAll)  # validate processing
            if max_mae > mae:
                print('Val loss decrease from {:.3f} to {:.3f}, '
                      'saving to {}, in epoch {:d}'.format(max_mae, mae, self.hp.save_path, epoch))
                max_mae = mae
                self.saver.save(self.sess, save_path=self.hp.save_path)

    def evaluate(self, testX, testDoW, testD, testH, testM, testL, testXAll):
        '''
        :return:
        '''
        labels_list, pres_list = list(), list()
        if not self.is_training:
            # model_file = tf.train.latest_checkpoint(self.para.save_path)
            saver = tf.train.import_meta_graph(self.hp.save_path + '.meta')
            # saver.restore(sess, model_file)
            print('the model weights has been loaded from the path of {:s}.'.format(self.hp.save_path))
            saver.restore(self.sess, self.hp.save_path)
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
            xs = np.expand_dims(testX[start_idx: end_idx], axis=-1)
            day_of_week = np.reshape(testDoW[start_idx: end_idx], [-1, self.site_num])
            minute_of_day = np.reshape(testM[start_idx: end_idx], [-1, self.site_num])
            labels = testL[start_idx: end_idx]
            xs_all = np.expand_dims(testXAll[start_idx: end_idx], axis=-1)
            sp, dis = sp_dis(hp=self.hp)
            in_deg, out_deg = in_out_deg(self.hp)
            feed_dict = construct_feed_dict(xs, xs_all, self.adj, labels, day_of_week, minute_of_day, self.placeholders,
                                            site_num=self.site_num, sp=sp, dis=dis, in_deg=in_deg, out_deg=out_deg, is_training=False)
            feed_dict.update({self.placeholders['dropout']: 0.0})
            pres = self.sess.run((self.pre), feed_dict=feed_dict)

            labels_list.append(labels)
            pres_list.append(pres)

        end_time = datetime.datetime.now()
        total_time = end_time - start_time
        print("Total running times is : %f" % total_time.total_seconds())

        labels_list = np.concatenate(labels_list, axis=0)
        pres_list = np.concatenate(pres_list, axis=0)
        np.savez_compressed('data/MT-STNet-' + 'YINCHUAN', **{'prediction': pres_list, 'truth': labels_list})
        if not self.is_training:
            print('                MAE\t\tRMSE\t\tMAPE')
            for (l,r) in [(0,66)]:
                for i in range(self.output_len):
                    mae, rmse, mape = metric(pres_list[:,l:r,i], labels_list[:,l:r,i])
                    print('step: %02d         %.3f\t\t%.3f\t\t%.3f%%' % (i + 1, mae, rmse, mape * 100))
                mae, rmse, mape = metric(pres_list[:,l:r], labels_list[:,l:r])  # 产生预测指标
                print('average:         %.3f\t\t%.3f\t\t%.3f%%' %(mae, rmse, mape * 100))
                print('\n')
        mae, rmse, mape = metric(pres_list, labels_list)
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

    trainX, trainDoW, trainD, trainH, trainM, trainL, trainXAll, valX, valDoW, valD, valH, valM, valL, valXAll, testX, testDoW, testD, testH, testM, testL, testXAll, mean, std = loadData(para)
    print('trainX: %s\ttrainY: %s' % (trainX.shape, trainL.shape))
    print('valX:   %s\t\tvalY:   %s' % (valX.shape, valL.shape))
    print('testX:  %s\t\ttestY:  %s' % (testX.shape, testL.shape))
    print('data loaded!')

    pre_model = Model(para, mean=mean, std=std)
    pre_model.initialize_session(session)

    if int(val) == 1:
        pre_model.run_epoch(trainX, trainDoW, trainD, trainH, trainM, trainL, trainXAll, valX, valDoW, valD, valH, valM, valL, valXAll, testX, testDoW, testD, testH, testM, testL, testXAll)
    else:
        pre_model.evaluate(testX, testDoW, testD, testH, testM, testL, testXAll)

    print('#...................................finished............................................#')


if __name__ == '__main__':
    main()