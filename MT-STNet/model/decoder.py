# -- coding: utf-8 --
from model.spatial_attention import Transformer
import tensorflow as tf
from model.temporal_attention import t_attention


class Decoder_ST(object):
    def __init__(self, hp, placeholders=None, model_func=None):
        '''
        :param hp:
        '''
        self.hp = hp
        self.placeholders = placeholders
        self.model_func = model_func

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

    def decoder_spatio_temporal(self, features=None, day=None, hour=None, minute=None, position=None, supports=None, sp=None, dis=None,in_deg=None,out_deg=None):
        '''
        :param flow:
        :param day:
        :param hour:
        :param position:
        :return:
        '''
        pres = list()
        '''
        decoder_gcn = self.model_func(self.placeholders,
                                      input_dim=self.hp.emb_size,
                                      para=self.hp,
                                      supports=supports)
        '''
        m = Transformer(self.hp)
        features = tf.reshape(tf.transpose(features, perm=[0, 2, 1, 3]),
                              shape=[-1, self.hp.input_length, self.hp.emb_size])  # 3-D
        for i in range(self.hp.output_length):
            o_day = day[:, i:i + 1, :, :]
            o_hour = hour[:, i:i + 1, :, :]
            o_minute = minute[:, i:i + 1, :, :]

            pre_features = tf.add_n([o_day, o_hour, o_minute])
            pre_features = tf.reshape(tf.transpose(pre_features, perm=[0, 2, 1, 3]),
                                      shape=[-1, 1, self.hp.emb_size])  # 3-D

            print('in the decoder step, the input_features shape is : ', features.shape)
            print('in the decoder step, the pre_features shape is : ', pre_features.shape)

            # x = m.encoder(speed=features, day=day, hour=hour, minute=minute, position=position)
            t_features = t_attention(hiddens=features,
                                     hidden=pre_features,
                                     hidden_units=self.hp.emb_size,
                                     dropout_rate=self.hp.dropout,
                                     is_training=self.hp.is_training)  # temporal attention, shape is [-1, length, hidden_size]
            # ,num_heads=self.hp.num_heads,num_blocks=self.hp.num_blocks
            features = tf.concat([features, t_features], axis=1)

            # x = m.encoder(inputs=t_features,
            #               input_length=1,
            #               day=o_day,
            #               hour=o_hour,
            #               minute=o_minute,
            #               position=position)  # spatial attention
            x = tf.squeeze(t_features)
            x = tf.reshape(x, shape=[-1, self.hp.site_num, self.hp.emb_size])
            results = tf.layers.dense(inputs=x, units=1, name='layer', reuse=tf.AUTO_REUSE)
            pre = tf.reshape(results, shape=[-1, self.hp.site_num])

            # to store the prediction results for road nodes on each time
            pres.append(tf.expand_dims(pre, axis=-1))

        return tf.concat(pres, axis=-1, name='output_y')

    def decoder_spatio_temporal_1(self, features=None, day=None, hour=None, minute=None, position=None, supports=None,
                                  in_length=6, sp=None, dis=None,in_deg=None,out_deg=None):
        '''
        :param flow:
        :param day:
        :param hour:
        :param position:
        :return:
        '''
        x_0 = tf.Variable(tf.truncated_normal(shape=[1, 1, 1, self.hp.emb_size]), name='w_1')
        pre_features = tf.add_n([hour, minute])
        pre_features = pre_features + x_0
        pre_features = tf.reshape(tf.transpose(pre_features, perm=[0, 2, 1, 3]),
                                  shape=[-1, self.hp.output_length, self.hp.emb_size])  # 3-D
        # features = tf.reshape(features, shape=[self.hp.batch_size, self.hp.input_length, self.hp.site_num, self.hp.emb_size])
        features = tf.reshape(tf.transpose(features, perm=[0, 2, 1, 3]),
                              shape=[-1, self.hp.input_length, self.hp.emb_size])
        in_features = features

        # with tf.variable_scope("temporal_attention_1"):
        x = t_attention(hiddens=features, hidden=pre_features, hidden_units=self.hp.emb_size,
                        dropout_rate=self.hp.dropout, is_training=self.hp.is_training)  # temporal attention
        x = tf.reshape(x, shape=[-1, self.hp.site_num, self.hp.output_length, self.hp.emb_size])
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        features = tf.reshape(x, shape=[-1, self.hp.site_num, self.hp.emb_size])

        m = Transformer(self.hp)
        x = m.encoder(inputs=features,
                      input_length=self.hp.output_length,
                      day=day,
                      hour=hour,
                      minute=minute,
                      position=position,
                      sp=sp,
                      dis=dis,
                      in_deg=in_deg,
                      out_deg=out_deg)  # spatial attention

        features = tf.concat(tf.split(features, self.hp.num_heads, axis=2), axis=0)
        encoder_gcn = self.model_func(placeholders=self.placeholders,
                                      input_dim=self.hp.emb_size // self.hp.num_heads,
                                      para=self.hp,
                                      supports=supports)
        encoder_gcn_out = encoder_gcn.predict(features)
        encoder_gcn_out = tf.concat(tf.split(encoder_gcn_out, self.hp.num_heads, axis=0), axis=2)

        encoder_out = tf.concat([x, encoder_gcn_out], axis=-1)
        encoder_out = tf.layers.dense(encoder_out, units=self.hp.emb_size, activation=tf.nn.tanh, name='concate')
        # encoder_out=self.gate_fusion(states=x,inputs=encoder_gcn_out,hidden_size=self.hp.emb_size)

        # bias = tf.Variable(tf.truncated_normal(shape=[self.hp.emb_size]), name='bias')
        # z = tf.nn.sigmoid(tf.add(tf.add_n([tf.layers.dense(x, self.hp.emb_size,
        #                                                    kernel_initializer=tf.truncated_normal_initializer(
        #                                                        dtype=tf.float32), name='w_1'),
        #                                    tf.layers.dense(x, self.hp.emb_size,
        #                                                    kernel_initializer=tf.truncated_normal_initializer(
        #                                                        dtype=tf.float32), name='w_2')]), bias))
        # # z = tf.nn.sigmoid(tf.add(tf.add_n([tf.matmul(x,w_1), tf.matmul(encoder_gcn_out,w_2)]),bias))
        # encoder_out = tf.multiply(z, x) + tf.multiply(1 - z, encoder_gcn_out)

        # encoder_out = tf.concat([x,encoder_gcn_out],axis=-1)
        # encoder_out=pre_features

        encoder_out = tf.reshape(encoder_out,
                                 shape=[self.hp.batch_size, self.hp.output_length, self.hp.site_num, self.hp.emb_size])
        encoder_out = tf.transpose(encoder_out, [0, 2, 1, 3])
        encoder_out = tf.reshape(encoder_out, shape=[-1, self.hp.output_length, self.hp.emb_size])

        # 输入和输出合并
        in_features = tf.concat([in_features, encoder_out], axis=1)

        # with tf.variable_scope("temporal_attention_2"):
        encoder_out = t_attention(hiddens=in_features, hidden=encoder_out, hidden_units=self.hp.emb_size,
                                  dropout_rate=self.hp.dropout, is_training=self.hp.is_training)
        encoder_out = tf.reshape(encoder_out,
                                 shape=[self.hp.batch_size, self.hp.site_num, self.hp.output_length, self.hp.emb_size])
        if self.hp.model_name=='MT_STNet':
            results_1 = tf.layers.dense(inputs=encoder_out[:, 0:13, :, :], units=64, name='task_1')
            results_1 = tf.layers.dense(inputs=results_1, units=1, name='task_1_1')
            results_2 = tf.layers.dense(inputs=encoder_out[:, 13: 26, :, :], units=64, name='task_2')
            results_2 = tf.layers.dense(inputs=results_2, units=1, name='task_2_1')
            results_3 = tf.layers.dense(inputs=encoder_out[:, 26:, :, :], units=64, name='task_3')
            results_3 = tf.layers.dense(inputs=results_3, units=1, name='task_3_1')
            results = tf.concat([results_1, results_2, results_3], axis=1)
            results = tf.squeeze(results, axis=-1, name='output_y')
        else:
            results = tf.layers.dense(inputs=encoder_out, units=128, name='layer_1', reuse=tf.AUTO_REUSE)
            results = tf.layers.dense(inputs=results, units=1, name='layer_2', reuse=tf.AUTO_REUSE)
            results = tf.squeeze(results, axis=-1, name='output_y')
        return results