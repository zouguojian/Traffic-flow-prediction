# -- coding: utf-8 --
from model.utils import *
from model.inits import *
from model.models import GCN
from model import tf_utils

def fusionGate(x, y):
    '''
    :param x: [-1, len, site, dim]
    :param y: [-1, len, site, dim]
    :return: [-1, len, site, dim]
    '''
    z = tf.nn.sigmoid(tf.multiply(x, y))
    h = tf.add(tf.multiply(z, x), tf.multiply(1 - z, y))
    return h

def featureGate(X, Y, K, d, bn, bn_decay, is_training):
    '''
    :param x: [-1, len, site, dim]
    :param y: [-1, len, site, dim]
    :return: [-1, len, site, dim]
    '''
    D = K*d
    X = FC(
        X, units=D, activations=None,
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    Y = FC(
        Y, units=D, activations=None,
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    h = tf.nn.tanh(tf.add(X, Y))
    return h

def spatialAttention(X, STE, K, d, bn, bn_decay, is_training, model_name='MT-STNet', spatial_inf=None, hp=None):
    '''
    spatial attention mechanism
    X:      [batch_size, num_step, N, D]
    STE:    [batch_size, num_step, N, D]
    K:      number of attention heads
    d:      dimension of each attention outputs
    return: [batch_size, num_step, N, D]
    '''
    X = tf.add(X, STE)
    if hp.is_physical:
        in_deg, out_deg = spatial_inf[2], spatial_inf[3]
        deg_emb = tf.add(in_deg, out_deg)
        X = tf.add(X, deg_emb)
    D = K * d
    query = FC(
        X, units=D, activations=tf.nn.relu,
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    key = FC(
        X, units=D, activations=tf.nn.relu,
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    value = FC(
        X, units=D, activations=tf.nn.relu,
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    # [K * batch_size, num_step, N, d]
    query = tf.concat(tf.split(query, K, axis=-1), axis=0)
    key = tf.concat(tf.split(key, K, axis=-1), axis=0)
    value = tf.concat(tf.split(value, K, axis=-1), axis=0)

    # [K * batch_size, num_step, N, N]
    outputs = tf.matmul(query, key, transpose_b=True)
    outputs /= (d ** 0.5)

    if hp.is_physical:
        dis, sp = spatial_inf[0],spatial_inf[1]
        dis = tf.expand_dims(tf.expand_dims(dis, axis=0),axis=0) # [1, 1, N, N]
        outputs = tf.add(outputs, dis)  # physical information

        sp = tf.reshape(sp, shape=[outputs.shape[-1], outputs.shape[-1], sp.shape[1], sp.shape[2]]) # [N, N, max_len=15, dim]
        sp = tf.expand_dims(sp, axis=0) # [1, N, N, max_len=15, dim]
        with tf.variable_scope('w_n', reuse=tf.AUTO_REUSE):
            w_n = tf.get_variable(initializer=tf.glorot_normal_initializer()(shape=[1, 1, sp.shape[1], K*d, 1]), dtype=tf.float32,
                                     trainable=True, name='w')
        sp = tf.matmul(sp, w_n)
        sp = tf.squeeze(sp, axis=-1)
        sp = tf.reduce_mean(sp, axis=-1)
        sp = tf.expand_dims(sp,axis=1) # [1, 1, N, N]
        outputs = tf.add(outputs, sp)  # physical information

    attention = tf.nn.softmax(outputs, axis=-1)
    # [batch_size, num_step, N, D]
    X = tf.matmul(attention, value)
    X = tf.concat(tf.split(X, K, axis=0), axis=-1)
    X = FC(
        X, units=[D, D], activations=[tf.nn.relu, None],
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    return X

def temporalAttention(X, STE, K, d, bn, bn_decay, is_training, mask=True, model_name='MT-STNet',spatial_inf=None, hp=None):
    '''
    temporal attention mechanism
    X:      [batch_size, num_step, N, D]
    STE:    [batch_size, num_step, N, D]
    K:      number of attention heads
    d:      dimension of each attention outputs
    return: [batch_size, num_step, N, D]
    '''
    X = tf.add(X, STE)
    if hp.is_physical:
        in_deg, out_deg = spatial_inf[2], spatial_inf[3]
        deg_emb = tf.add(in_deg, out_deg)
        X = tf.add(X, deg_emb)
    D = K * d
    query = FC(
        X, units=D, activations=tf.nn.relu,
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    key = FC(
        X, units=D, activations=tf.nn.relu,
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    value = FC(
        X, units=D, activations=tf.nn.relu,
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    # [K * batch_size, num_step, N, d]
    query = tf.concat(tf.split(query, K, axis=-1), axis=0)
    key = tf.concat(tf.split(key, K, axis=-1), axis=0)
    value = tf.concat(tf.split(value, K, axis=-1), axis=0)
    # query: [K * batch_size, N, num_step, d]
    # key:   [K * batch_size, N, d, num_step]
    # value: [K * batch_size, N, num_step, d]
    query = tf.transpose(query, perm=(0, 2, 1, 3))
    key = tf.transpose(key, perm=(0, 2, 3, 1))
    value = tf.transpose(value, perm=(0, 2, 1, 3))
    # [K * batch_size, N, num_step, num_step]
    attention = tf.matmul(query, key)
    attention /= (d ** 0.5)
    # mask attention score
    if mask:
        batch_size = tf.shape(X)[0]
        num_step = X.get_shape()[1].value
        N = X.get_shape()[2].value
        mask = tf.ones(shape=(num_step, num_step))
        mask = tf.linalg.LinearOperatorLowerTriangular(mask).to_dense()
        mask = tf.expand_dims(tf.expand_dims(mask, axis=0), axis=0)
        mask = tf.tile(mask, multiples=(K * batch_size, N, 1, 1))
        mask = tf.cast(mask, dtype=tf.bool)
        attention = tf.compat.v2.where(
            condition=mask, x=attention, y=-2 ** 15 + 1)
    # softmax
    attention = tf.nn.softmax(attention, axis=-1)
    # [batch_size, num_step, N, D]
    X = tf.matmul(attention, value)
    X = tf.transpose(X, perm=(0, 2, 1, 3))
    X = tf.concat(tf.split(X, K, axis=0), axis=-1)
    X = FC(
        X, units=[D, D], activations=[tf.nn.relu, None],
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    return X



def BridgeTrans(X, STE_P, STE_Q, K, d, bn, bn_decay, is_training, spatial_inf=None):
    '''
    transform attention mechanism
    X:      [batch_size, P, N, D]
    STE_P:  [batch_size, P, N, D]
    STE_Q:  [batch_size, Q, N, D]
    K:      number of attention heads
    d:      dimension of each attention outputs
    return: [batch_size, Q, N, D]
    '''
    D = K * d
    # query: [batch_size, Q, N, K * d]
    # key:   [batch_size, P, N, K * d]
    # value: [batch_size, P, N, K * d]
    query = tf_utils.FC(
                        STE_Q, units=D, activations=tf.nn.relu,
                        bn=bn, bn_decay=bn_decay, is_training=is_training, name='bridge1')
    key = tf_utils.FC(
                        STE_P, units=D, activations=tf.nn.relu,
                        bn=bn, bn_decay=bn_decay, is_training=is_training, name='bridge2')
    value = tf_utils.FC(
                        X, units=D, activations=tf.nn.relu,
                        bn=bn, bn_decay=bn_decay, is_training=is_training, name='bridge3')
    # query: [K * batch_size, Q, N, d]
    # key:   [K * batch_size, P, N, d]
    # value: [K * batch_size, P, N, d]
    query = tf.concat(tf.split(query, K, axis=-1), axis=0)
    key = tf.concat(tf.split(key, K, axis=-1), axis=0)
    value = tf.concat(tf.split(value, K, axis=-1), axis=0)
    # query: [K * batch_size, N, Q, d]
    # key:   [K * batch_size, N, d, P]
    # value: [K * batch_size, N, P, d]
    query = tf.transpose(query, perm=(0, 2, 1, 3))
    key = tf.transpose(key, perm=(0, 2, 3, 1))
    value = tf.transpose(value, perm=(0, 2, 1, 3))
    # [K * batch_size, N, Q, P]
    attention = tf.matmul(query, key)
    attention /= (d ** 0.5)
    attention = tf.nn.softmax(attention, axis=-1)
    # [batch_size, Q, N, D]
    X = tf.matmul(attention, value)
    X = tf.transpose(X, perm=(0, 2, 1, 3))
    X = tf.concat(tf.split(X, K, axis=0), axis=-1)
    X = tf_utils.FC(
        X, units=[D, D], activations=[tf.nn.relu, None],
        bn=bn, bn_decay=bn_decay, is_training=is_training, name='bridge4')
    return X


# def BridgeTrans(X, STE_P, STE_Q, K, d, bn, bn_decay, is_training, spatial_inf=None):
#     '''
#     transform attention mechanism
#     X:      [batch_size, P, N, D]
#     STE_P:  [batch_size, P, N, D]
#     STE_Q:  [batch_size, Q, N, D]
#     K:      number of attention heads
#     d:      dimension of each attention outputs
#     return: [batch_size, Q, N, D]
#     '''
#     D = K * d
#     # query: [batch_size, Q, N, K * d]
#     # key:   [batch_size, P, N, K * d]
#     # value: [batch_size, P, N, K * d]
#     query = FC(
#         STE_Q, units=D, activations=tf.nn.relu,
#         bn=bn, bn_decay=bn_decay, is_training=is_training)
#     key = FC(
#         STE_P, units=D, activations=tf.nn.relu,
#         bn=bn, bn_decay=bn_decay, is_training=is_training)
#     value = FC(
#         X, units=D, activations=tf.nn.relu,
#         bn=bn, bn_decay=bn_decay, is_training=is_training)
#     # query: [K * batch_size, Q, N, d]
#     # key:   [K * batch_size, P, N, d]
#     # value: [K * batch_size, P, N, d]
#     query = tf.concat(tf.split(query, K, axis=-1), axis=0)
#     key = tf.concat(tf.split(key, K, axis=-1), axis=0)
#     value = tf.concat(tf.split(value, K, axis=-1), axis=0)
#     # query: [K * batch_size, N, Q, d]
#     # key:   [K * batch_size, N, d, P]
#     # value: [K * batch_size, N, P, d]
#     query = tf.transpose(query, perm=(0, 2, 1, 3))
#     key = tf.transpose(key, perm=(0, 2, 3, 1))
#     value = tf.transpose(value, perm=(0, 2, 1, 3))
#     # [K * batch_size, N, Q, P]
#     attention = tf.matmul(query, key)
#     attention /= (d ** 0.5)
#     attention = tf.nn.softmax(attention, axis=-1)
#     # [batch_size, Q, N, D]
#     X = tf.matmul(attention, value)
#     X = tf.transpose(X, perm=(0, 2, 1, 3))
#     X = tf.concat(tf.split(X, K, axis=0), axis=-1)
#     X = FC(
#         X, units=[D, D], activations=[tf.nn.relu, None],
#         bn=bn, bn_decay=bn_decay, is_training=is_training)
#     return X

def STAttBlock(X, STE, K, d, bn, bn_decay, is_training, mask=True, placeholders=None, hp=None, supports=None, model_name='MT-STNet', spatial_inf=None):
    HT = temporalAttention(X, STE, K, d, bn, bn_decay, is_training, mask=mask, model_name=model_name, spatial_inf=spatial_inf, hp=hp)

    shape = HT.get_shape().as_list()
    encoder_gcn = GCN(placeholders=placeholders,
                             input_dim= shape[3],
                             para= hp,
                             supports=supports)
    GS = tf.reshape(HT, [-1, shape[2], shape[3]])
    GS = encoder_gcn.predict(GS)
    GS = tf.reshape(GS, [-1, shape[1], shape[2], shape[3]])

    HS = spatialAttention(HT, STE, K, d, bn, bn_decay, is_training, model_name=model_name, spatial_inf=spatial_inf, hp=hp)
    H = fusionGate(HS, GS)
    # H = featureGate(HS, GS, K, d, bn, bn_decay, is_training)
    return tf.add(X, H)

def MT_STNet(X, X_all, TE, SE, P, Q, S, L, K, d, bn, bn_decay, is_training, supports, placeholders, spatial_inf, hp, model_name):
    '''
    section of encoder, --- for example, inputs.shape is :  (32, 12, 66, 32)
    '''
    STE = STEmbedding(SE= SE,
                        TE=TE,
                        D = K * d,
                        bn=bn,
                        bn_decay=bn_decay,
                        is_training=is_training)
    X_Q = FC(
        X_all[:,P:], units=[K * d, K * d], activations=[tf.nn.relu, None],
        bn=bn, bn_decay=bn_decay, is_training=is_training)  # [-1, P+Q, N, dim]
    # STE = tf.add(STE, X_all)
    STE_P = STE[:, : P]
    STE_Q = STE[:, P :]
    X = FC(
        tf.concat([X,X_all[:,:P]],axis=-1), units=[K * d, K * d], activations=[tf.nn.relu, None],
        bn=bn, bn_decay=bn_decay, is_training=is_training)  # [-1, P, N, dim]
    
    for i in range(L):
        X = STAttBlock(X=X,
                        STE=STE_P,
                        K=K,
                        d= d,
                        bn=bn,
                        bn_decay=bn_decay,
                        is_training=is_training,
                        mask=False,
                        placeholders=placeholders,
                        hp=hp,
                        supports= supports,
                        model_name=model_name,
                        spatial_inf=spatial_inf)
    X_E = X
    print('encoder\'s output shape is ',X.shape)
    '''
    section of decoder --- for example, inputs.shape is :  (32, 18, 66, 32)
    '''
    for i in range(hp.num_blocks):
        X = STAttBlock(X= tf.concat([X[:,-S:],X_Q],axis=1),
                        STE=STE[:,-(Q + S):],
                        K=K,
                        d= d,
                        bn=bn,
                        bn_decay=bn_decay,
                        is_training=is_training,
                        mask=True,
                        placeholders=placeholders,
                        hp=hp,
                        supports=supports,
                        model_name=model_name,
                        spatial_inf=spatial_inf)
    X_D = X[:,-Q:]
    print('decoder\'s output shape is ', X_D.shape)
    '''
    section of generative --- for example, inputs.shape is :  (32, 12, 66, 32), (32, 12, 66, 32), 
    '''
    # X = BridgeTrans(X=X_E,
    #                 STE_P=tf.add(X_E, STE_P),
    #                 STE_Q=tf.add(X_D, STE_Q),
    #                 K=K,
    #                 d=d,
    #                 bn=bn,
    #                 bn_decay=bn_decay,
    #                 is_training=is_training)
    # print('bridge\'s output shape is ', X.shape)

    for i in range(Q):
        X = BridgeTrans(X=X_E,
                        STE_P= X_E,
                        STE_Q=tf.add(X_E[:,-1:], STE_Q[:,i:i+1]),
                        K=K,
                        d=d,
                        bn=bn,
                        bn_decay=bn_decay,
                        is_training=is_training)
        X_E = tf.concat([X_E, tf.add(X, STE_Q[:,i:i+1])], axis=1)
    X = X_E[:,-Q:]
    print('bridge\'s output shape is ', X.shape)

    '''
    section of generative inference --- for example, inputs.shape is :  (32, 12, 162, 32)
    '''
    pre1 = FC(
            X[:,:,:13], units=[K * d, 1], activations=[tf.nn.relu, None],
            bn=bn, bn_decay=bn_decay, is_training=is_training,
            use_bias=True, drop=0.1)
    pre2 = FC(
        X[:,:,13:26], units=[K * d, 1], activations=[tf.nn.relu, None],
        bn=bn, bn_decay=bn_decay, is_training=is_training,
        use_bias=True, drop=0.1)
    pre3 = FC(
        X[:,:,26:], units=[K * d, 1], activations=[tf.nn.relu, None],
        bn=bn, bn_decay=bn_decay, is_training=is_training,
        use_bias=True, drop=0.1)
    pre = tf.squeeze(tf.concat([pre1, pre2, pre3],axis=2), axis=3)
    return pre