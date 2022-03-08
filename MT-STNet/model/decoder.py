# -- coding: utf-8 --
from model.spatial_attention import Transformer
import tensorflow as tf
from model.temporal_attention import t_attention

class Decoder_ST(object):
    def __init__(self, hp, placeholders=None, model_func=None):
        '''
        :param hp:
        '''
        self.hp=hp
        self.placeholders=placeholders
        self.model_func=model_func

    def attention(self, h_t, encoder_hs):
        '''
        h_t for decoder, the shape is [batch, 1, h]
        encoder_hs for encoder, the shape is [batch, time ,h]
        :param h_t:
        :param encoder_hs:
        :return: [None, hidden size]
        '''
        scores = tf.reduce_sum(tf.multiply(encoder_hs, tf.tile(h_t,multiples=[1,encoder_hs.shape[1],1])), 2)
        a_t = tf.nn.softmax(scores)  # [batch, time]
        a_t = tf.expand_dims(a_t, 2) # [batch, time, 1]
        c_t = tf.matmul(tf.transpose(encoder_hs, perm=[0,2,1]), a_t) #[batch ,h , 1]
        c_t = tf.squeeze(c_t, axis=2) #[batch, h]]
        h_t=tf.squeeze(h_t,axis=1)
        h_tld  = tf.layers.dense(tf.concat([h_t, c_t], axis=1),units=c_t.shape[-1],activation=tf.nn.relu) #[batch, h]
        return h_tld

    def decoding(self, encoder_hs, gcn=None, gan=None, site_num=None, x_p=None, day=None, hour=None, position=None):
        '''
        :param encoder_hs: [batch, time ,site num, hidden size]
        :param gcn:
        :param site_num:
        :return: [batch, site num, prediction size], [batch, prediction size]
        '''
        pres = list()
        shape=encoder_hs.shape
        h_states=encoder_hs[:,-1,:,:]
        encoder_hs = tf.reshape(tf.transpose(encoder_hs, perm=[0, 2, 1, 3]),shape=[shape[0] * shape[2], shape[1], shape[3]])

        for i in range(self.hp.output_length):
            # gcn for decoder processing, there is no question
            out_day=day[:,i,:,:]
            out_hour=hour[:,i,:,:]
            h_states = tf.layers.dense(inputs=h_states, units=out_day.shape[-1], reuse=tf.AUTO_REUSE)
            features=tf.add_n([h_states,out_day,out_hour,position[:,-1,:,:]])

            gcn_outs = gcn.predict(features) # gcn

            gan.input_length=1
            x = gan.encoder(speed=h_states, day=out_day, hour=out_hour, position=position[:,-1,:,:]) # gan

            features=tf.add_n([gcn_outs, x, position[:,-1,:,:]])
            features = tf.reshape(features, shape=[self.hp.batch_size, 1, features.shape[-1]])

            print('features shape is : ',features.shape)

            h_state, state = None, None
            initial_state = state

            # compute the attention state
            h_state = t_attention(hiddens=encoder_hs, hidden=h_state, hidden_units=shape[-1])  # attention # 注意修改
            # h_state = self.attention(h_t=h_state, encoder_hs=encoder_hs)  # attention
            h_states=tf.reshape(h_state,shape=[-1,site_num,128])

            results = tf.layers.dense(inputs=h_state, units=1, name='layer', reuse=tf.AUTO_REUSE, activation=tf.nn.relu)
            pre=tf.reshape(results,shape=[-1,site_num])
            # to store the prediction results for road nodes on each time
            pres.append(tf.expand_dims(pre, axis=-1))

        return tf.concat(pres, axis=-1,name='output_y')

    def decoder_spatio_temporal(self, features=None, day=None, hour=None, minute=None, position=None, supports=None):
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
        features = tf.reshape(tf.transpose(features, perm=[0, 2, 1, 3]), shape=[-1, self.hp.input_length, self.hp.emb_size])  # 3-D
        for i in range(self.hp.output_length):
            o_day = day[:, i:i+1, :, :]
            o_hour = hour[:, i:i+1, :, :]
            o_minute = minute[:, i:1+1, :, :]

            pre_features=tf.add_n([o_day, o_hour, o_minute])
            pre_features=tf.reshape(tf.transpose(pre_features, perm=[0, 2, 1, 3]),shape=[-1, 1, self.hp.emb_size]) #3-D

            print('in the decoder step, the input_features shape is : ', features.shape)
            print('in the decoder step, the pre_features shape is : ', pre_features.shape)

            # x = m.encoder(speed=features, day=day, hour=hour, minute=minute, position=position)
            t_features = t_attention(hiddens=features,
                                     hidden=pre_features,
                                     hidden_units=self.hp.emb_size,
                                     dropout_rate = self.hp.dropout,
                                     is_training=self.hp.is_training)  # temporal attention, shape is [-1, length, hidden_size]
            # ,num_heads=self.hp.num_heads,num_blocks=self.hp.num_blocks
            features = tf.concat([features, t_features], axis=1)

            x = m.encoder(inputs=t_features,
                          input_length=1,
                          day=o_day,
                          hour=o_hour,
                          minute=o_minute,
                          position=position)  # spatial attention
            x = tf.squeeze(x)
            x=tf.reshape(x,shape=[-1, self.hp.site_num, self.hp.emb_size])
            results = tf.layers.dense(inputs=x, units=1, name='layer', reuse=tf.AUTO_REUSE)
            pre=tf.reshape(results,shape=[-1,self.hp.site_num])

            # to store the prediction results for road nodes on each time
            pres.append(tf.expand_dims(pre, axis=-1))

        return tf.concat(pres, axis=-1, name='output_y')