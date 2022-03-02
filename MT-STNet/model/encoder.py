# -- coding: utf-8 --
from model.gat import Transformer
import tensorflow as tf
from model.temporal_attention import t_attention

class Encoder_ST(object):
    def __init__(self, hp, placeholders=None, model_func=None):
        '''
        :param hp:
        '''
        self.hp=hp
        self.placeholders=placeholders
        self.model_func=model_func

    def encoder_spatio_temporal(self, features=None, day=None, hour=None, minute=None, position=None, supports=None):
        '''
        :param flow:
        :param day:
        :param hour:
        :param position:
        :return:
        '''

        x = tf.reshape(features, shape=[self.hp.batch_size, self.hp.input_length, self.hp.site_num, self.hp.emb_size])
        x = tf.reshape(tf.transpose(x, perm=[0, 2, 1, 3]),shape=[-1, self.hp.input_length, self.hp.emb_size])
        x = t_attention(hiddens=x, hidden=x, hidden_units=self.hp.emb_size,num_heads=self.hp.num_heads,num_blocks=self.hp.num_blocks) # temporal attention
        x = tf.reshape(x, shape=[-1, self.hp.site_num, self.hp.input_length, self.hp.emb_size])
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        features = tf.reshape(x, shape=[-1, self.hp.site_num, self.hp.emb_size])

        m = Transformer(self.hp)
        x = m.encoder(speed=features, day=day, hour=hour, minute=minute, position=position) # spatial attention

        '''
        features=tf.add_n(inputs=[features,
                                  tf.reshape(day,[-1,self.hp.site_num, self.hp.emb_size]),
                                  tf.reshape(hour,[-1,self.hp.site_num, self.hp.emb_size])])

        encoder_gcn = self.model_func(self.placeholders,
                                      input_dim=self.hp.emb_size,
                                      para=self.hp,
                                      supports=supports)
        encoder_outs = encoder_gcn.predict(features)
        encoder_outs = tf.reshape(encoder_outs, shape=[self.hp.batch_size,
                                                       self.hp.input_length,
                                                       self.hp.site_num,
                                                       self.hp.gcn_output_size])
        print('encoder gcn outs shape is : ', encoder_outs.shape)
        '''

        x = tf.reshape(x, shape=[self.hp.batch_size, self.hp.input_length, self.hp.site_num, self.hp.emb_size])
        # trick
        # encoder_out = tf.add_n([x, encoder_outs, self.p_emd])
        encoder_out = x
        return encoder_out