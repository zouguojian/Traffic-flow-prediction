# -- coding: utf-8 --
from utils import *


class DelaClass(object):
    def __init__(self, hp, placeholders):
        self.hp = hp
        self.hidden_size = self.hp.hidden_size
        self.layer_num = self.hp.hidden_layer
        self.placeholders = placeholders
        self.h = 3
        self.w = 3
        self.input_length = self.hp.input_length
        self.output_length = self.hp.output_length
        self.site_num = self.hp.site_num
        self.features = self.hp.features

    def attention(self, h_t, encoder_hs):
        '''
        h_t for decoder, the shape is [batch, 1, h]
        encoder_hs for encoder, the shape is [batch, time ,h]
        :param h_t:
        :param encoder_hs:
        :return: [None, hidden size]
        '''
        scores = tf.reduce_sum(tf.multiply(encoder_hs, tf.tile(h_t, multiples=[1, encoder_hs.shape[1], 1])), 2)
        a_t = tf.nn.softmax(scores)  # [batch, time]
        a_t = tf.expand_dims(a_t, 2)  # [batch, time, 1]
        c_t = tf.matmul(tf.transpose(encoder_hs, perm=[0, 2, 1]), a_t)  # [batch ,h , 1]
        c_t = tf.squeeze(c_t, axis=2)  # [batch, h]]
        h_t = tf.squeeze(h_t, axis=1)
        h_tld = tf.layers.dense(tf.concat([h_t, c_t], axis=1), units=c_t.shape[-1], activation=tf.nn.relu)  # [batch, h]
        return h_tld

    def cnn(self, x=None):
        '''
        :param x: shape is [batch size,  input length, site num, features]
        :return: shape is [batch size, site num, hidden size]
        '''
        x = tf.transpose(x, perm=[0, 2, 1, 3])  # [batch size,  site num, input length, features]

        filter1 = tf.get_variable("filter1", [self.h, self.w, 1, 64],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        bias1 = tf.get_variable("bias1", [64], initializer=tf.truncated_normal_initializer(stddev=0.1))
        layer1 = tf.nn.conv2d(input=x, filter=filter1, strides=[1, 1, 1, 1], padding='SAME')
        # layer1=tf.layers.conv2d(inputs=x,filters=64,kernel_size=[3,3],padding='same',kernel_initializer=tf.truncated_normal_initializer())
        # bn1=tf.layers.batch_normalization(layer1,training=self.placeholders['is_training'])
        x = tf.add(layer1, bias1)
        relu1 = tf.nn.relu(layer1)

        filter2 = tf.get_variable("filter2", [self.h, self.w, 64, 64],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        bias2 = tf.get_variable("bias2", [64], initializer=tf.truncated_normal_initializer(stddev=0.1))
        layer2 = tf.nn.conv2d(input=relu1, filter=filter2, strides=[1, 1, 1, 1], padding='SAME')
        # bn2=tf.layers.batch_normalization(layer2,training=self.placeholders['is_training'])
        x = tf.add(layer2, bias2)
        relu2 = tf.nn.relu(x)

        filter3 = tf.get_variable("filter3", [self.h, self.w, 64, 64],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        bias3 = tf.get_variable("bias3", [64], initializer=tf.truncated_normal_initializer(stddev=0.1))
        layer3 = tf.nn.conv2d(input=relu2, filter=filter3, strides=[1, 1, 1, 1], padding='SAME')
        # bn3=tf.layers.batch_normalization(layer3,training=self.placeholders['is_training'])
        x = tf.add(layer3, bias3)
        relu3 = tf.nn.relu(x)

        cnn_x = tf.reduce_max(relu3, axis=2)

        return cnn_x

    def lstm(self):
        def cell():
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.hidden_size)
            lstm_cell_ = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell,
                                                       output_keep_prob=1 - self.placeholders['dropout'])
            return lstm_cell_

        mlstm = tf.nn.rnn_cell.MultiRNNCell([cell() for _ in range(self.layer_num)])
        return mlstm

    def encoding(self, x=None):
        # x = tf.layers.dense(inputs=x, units=self.hidden_size, name='dense_layer')
        cnn_x = self.cnn(x)

        x = tf.transpose(x, [0, 2, 1, 3])
        x = tf.reshape(x, shape=[-1, self.input_length, self.features])

        with tf.variable_scope('encoder_lstm'):
            lstm_x, _ = tf.nn.dynamic_rnn(cell=self.lstm(), inputs=x, dtype=tf.float32)
        lstm_x = tf.reshape(lstm_x[:, -1, :], shape=[-1, self.site_num, self.hidden_size])
        # hidden_x = tf.concat([cnn_x, lstm_x], axis=-1)
        hidden_x = [cnn_x, lstm_x]
        return hidden_x

    def decoding(self, x=None, embeddings=None, index_emb=None):
        embedding_x = tf.add_n(embeddings) + index_emb
        embedding_x = tf.reduce_sum(embedding_x,axis=1)
        # x = tf.concat([x, embedding_x],axis=-1)

        l_1 = tf.layers.dense(inputs=embedding_x, units=self.output_length, name='output_1')
        l_2 = tf.layers.dense(inputs=x[0], units=self.output_length, name='output_2')
        l_3 = tf.layers.dense(inputs=x[1], units=self.output_length, name='output_3')
        W = tf.get_variable("W", [1,1,1,3],initializer=tf.truncated_normal_initializer())
        pre = tf.multiply(tf.concat([tf.expand_dims(l_1,axis=-1),tf.expand_dims(l_2,axis=-1),tf.expand_dims(l_3,axis=-1)],axis=-1), W)
        pre = tf.reduce_mean(pre, axis=-1)
        return pre