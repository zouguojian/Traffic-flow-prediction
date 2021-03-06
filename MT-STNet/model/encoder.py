# -- coding: utf-8 --
from model.spatial_attention import Transformer
import tensorflow as tf
from model.temporal_attention import t_attention


class Encoder_ST(object):
    def __init__(self, hp, placeholders=None, model_func=None):
        '''
        :param hp:
        '''
        self.hp = hp
        self.placeholders = placeholders
        self.model_func = model_func

    def gate_attention(self, inputs, hidden_size):
        # inputs size是[batch_size, max_time, encoder_size(hidden_size)]
        u_context = tf.Variable(tf.truncated_normal([hidden_size]), name='u_context')
        # 使用一个全连接层编码GRU的输出的到期隐层表示,输出u的size是[batch_size, max_time, hidden_size]
        h = tf.layers.dense(inputs, hidden_size)
        # shape为[batch_size, max_time, 1]
        alpha = tf.nn.softmax(tf.reduce_sum(h, axis=2, keep_dims=True), dim=1)
        # alpha = tf.nn.softmax(tf.reduce_sum(tf.multiply(h, u_context), axis=2, keep_dims=True), dim=1)
        # reduce_sum之前shape为[batch_size, max_time, hidden_size]，之后shape为[batch_size, hidden_size]
        atten_output = tf.reduce_sum(tf.multiply(inputs, alpha), axis=1)
        return atten_output

    def gate_fusion(self, states=None, inputs=None, hidden_size=256):
        x = tf.concat([states, inputs], axis=-1)
        f = tf.layers.dense(x, units=hidden_size, activation=tf.nn.sigmoid,
                            kernel_initializer=tf.truncated_normal_initializer(dtype=tf.float32), name='forget',
                            reuse=tf.AUTO_REUSE)
        i = tf.layers.dense(x, units=hidden_size, activation=tf.nn.sigmoid,
                            kernel_initializer=tf.truncated_normal_initializer(dtype=tf.float32), name='inputs',
                            reuse=tf.AUTO_REUSE)
        c_t = tf.layers.dense(x, units=hidden_size, activation=tf.nn.tanh,
                              kernel_initializer=tf.truncated_normal_initializer(dtype=tf.float32), name='c_state',
                              reuse=tf.AUTO_REUSE)
        c = tf.multiply(f, states) + tf.multiply(i, c_t)
        return c

    def encoder_spatio_temporal(self, features=None, day=None, hour=None, minute=None, position=None, supports=None, sp=None, dis=None,in_deg=None,out_deg=None):
        '''
        :param flow:
        :param day:
        :param hour:
        :param position:
        :return:
        '''
        x = tf.reshape(features, shape=[self.hp.batch_size, self.hp.input_length, self.hp.site_num, self.hp.emb_size])
        date = tf.add_n([hour, minute])
        x = x + date
        # x = tf.concat([tf.expand_dims(tf.reshape(features,[-1, self.hp.emb_size]),axis=1),tf.expand_dims(tf.reshape(date, [-1, self.hp.emb_size]), axis=1)],axis=1)
        # x = tf.layers.conv1d(x,filters=self.hp.emb_size,kernel_size=2,strides=1,padding='valid')
        # print('after 1d convolutional operation, the x output shape is : ',x.shape)
        # x = tf.reshape(x, shape=[self.hp.batch_size, self.hp.input_length, self.hp.site_num, self.hp.emb_size])

        x = tf.reshape(tf.transpose(x, perm=[0, 2, 1, 3]), shape=[-1, self.hp.input_length, self.hp.emb_size])
        x = t_attention(hiddens=x, hidden=x, hidden_units=self.hp.emb_size, dropout_rate=self.hp.dropout,
                        is_training=self.hp.is_training)  # temporal attention
        # ,num_heads=self.hp.num_heads,num_blocks=self.hp.num_blocks
        x = tf.reshape(x, shape=[-1, self.hp.site_num, self.hp.input_length, self.hp.emb_size])
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        features = tf.reshape(x, shape=[-1, self.hp.site_num, self.hp.emb_size])
        if self.hp.model_name != 'STNet_4':
            m = Transformer(self.hp)
            x = m.encoder(inputs=features,
                          input_length=self.hp.input_length,
                          day=day,
                          hour=hour,
                          minute=minute,
                          position=position,
                          sp=sp,
                          dis=dis,
                          in_deg=in_deg,
                          out_deg=out_deg)  # spatial attention

        # features=tf.add_n(inputs=[features, tf.reshape(day,[-1,self.hp.site_num, self.hp.emb_size]),
        #                     tf.reshape(hour,[-1,self.hp.site_num, self.hp.emb_size]),
        #                     tf.reshape(minute,[-1,self.hp.site_num, self.hp.emb_size]),
        #                     tf.reshape(position,[-1,self.hp.site_num, self.hp.emb_size])])
        if self.hp.model_name != 'STNet_2':
            features = tf.concat(tf.split(features, self.hp.num_heads, axis=2), axis=0)
            encoder_gcn = self.model_func(placeholders=self.placeholders,
                                          input_dim=self.hp.emb_size // self.hp.num_heads,
                                          para=self.hp,
                                          supports=supports)
            encoder_gcn_out = encoder_gcn.predict(features)
            encoder_gcn_out = tf.concat(tf.split(encoder_gcn_out, self.hp.num_heads, axis=0), axis=2)
            if self.hp.model_name != 'STNet_4':
                encoder_out = tf.concat([x, encoder_gcn_out], axis=-1)
            else:encoder_out = encoder_gcn_out
            encoder_out = tf.layers.dense(encoder_out, units=self.hp.emb_size, activation=tf.nn.tanh, name='concate')
        else:encoder_out = x
        # encoder_out=self.gate_fusion(states=x,inputs=encoder_gcn_out,hidden_size=self.hp.emb_size)

        # w_1 = tf.Variable(tf.truncated_normal(shape=[self.hp.emb_size,self.hp.emb_size]), name='w_1')
        # w_2 = tf.Variable(tf.truncated_normal(shape=[self.hp.emb_size,self.hp.emb_size]), name='w_2')
        # bias = tf.Variable(tf.truncated_normal(shape=[self.hp.emb_size]), name='bias')
        # # z = tf.nn.sigmoid(tf.add(tf.add_n([tf.matmul(x,w_1), tf.matmul(encoder_gcn_out,w_2)]),bias))
        # z = tf.nn.sigmoid(tf.add(tf.add_n([tf.layers.dense(x,self.hp.emb_size,kernel_initializer=tf.truncated_normal_initializer(dtype=tf.float32),name='w_1'), tf.layers.dense(x,self.hp.emb_size,kernel_initializer=tf.truncated_normal_initializer(dtype=tf.float32),name='w_2')]),bias))
        # encoder_out = tf.multiply(z, x) + tf.multiply(1 - z, encoder_gcn_out)
        encoder_out = tf.reshape(encoder_out,
                                 shape=[self.hp.batch_size, self.hp.input_length, self.hp.site_num, self.hp.emb_size])
        # encoder_out = tf.reshape(x, shape=[self.hp.batch_size, self.hp.input_length, self.hp.site_num, self.hp.emb_size])

        # encoder_gcn_out = tf.reshape(encoder_gcn_out, shape=[self.hp.batch_size,
        #                                                      self.hp.input_length,
        #                                                      self.hp.site_num,
        #                                                      self.hp.emb_size])
        # print('encoder gcn out shape is : ', encoder_gcn_out.shape)
        # x = tf.reshape(x, shape=[self.hp.batch_size, self.hp.input_length, self.hp.site_num, self.hp.emb_size])

        # trick
        # encoder_out = tf.add_n([x, encoder_outs, self.p_emd])
        # encoder_out = x
        return encoder_out