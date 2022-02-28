# -- coding: utf-8 --
from model.gat import Transformer
import tensorflow as tf

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
        pres_p=list()
        shape=encoder_hs.shape
        h_states=encoder_hs[:,-1,:,:]
        encoder_hs = tf.reshape(tf.transpose(encoder_hs, perm=[0, 2, 1, 3]),shape=[shape[0] * shape[2], shape[1], shape[3]])
        initial_state = self.initial_state

        for i in range(self.predict_time):
            # gcn for decoder processing, there is no question
            out_day=day[:,i,:,:]
            out_hour=hour[:,i,:,:]
            h_states = tf.layers.dense(inputs=h_states, units=out_day.shape[-1], reuse=tf.AUTO_REUSE)
            features=tf.add_n([h_states,out_day,out_hour,position[:,-1,:,:]])

            gcn_outs = gcn.predict(features) # gcn

            gan.input_length=1
            x = gan.encoder(speed=h_states, day=out_day, hour=out_hour, position=position[:,-1,:,:]) # gan

            features=tf.add_n([gcn_outs, x, position[:,-1,:,:]])
            features = tf.reshape(features, shape=[self.batch_size, 1, features.shape[-1]])

            print('features shape is : ',features.shape)

            h_state, state = tf.nn.dynamic_rnn(cell=self.mlstm_cell,inputs=features, initial_state=initial_state, dtype=tf.float32)
            initial_state = state

            # compute the attention state
            h_state = T_attention(hiddens=encoder_hs, hidden=h_state, hidden_units=shape[-1])  # attention # 注意修改
            # h_state = self.attention(h_t=h_state, encoder_hs=encoder_hs)  # attention
            h_states=tf.reshape(h_state,shape=[-1,site_num,self.nodes])

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

        decoder_gcn = self.model_func(self.placeholders,
                                      input_dim=self.hp.emb_size,
                                      para=self.hp,
                                      supports=supports)
        m = Transformer(self.hp)
        x = m.encoder(speed=features, day=day, hour=hour, minute=minute, position=position)

        for i in range(self.hp.output_length):


        print('encoder output shape is : ', encoder_out.shape)