from model.inits import *

def FC(x, units, activations, bn, bn_decay, is_training, use_bias=True, drop=None, name=None):
    if isinstance(units, int):
        units = [units]
        activations = [activations]
    elif isinstance(units, tuple):
        units = list(units)
        activations = list(activations)
    assert type(units) == list
    i =1
    for num_unit, activation in zip(units, activations):
        if drop is not None:
            x = dropout(x, drop=drop, is_training=is_training)
        x = conv2d(
            x, output_dims=num_unit, kernel_size=[1, 1], stride=[1, 1],
            padding='VALID', use_bias=use_bias, activation=activation,
            bn=bn, bn_decay=bn_decay, is_training=is_training, name=name+str(i))
        i+=1
    return x

def conv2d(x, output_dims, kernel_size, stride = [1, 1],
           padding = 'SAME', use_bias = True, activation = tf.nn.relu,
           bn = False, bn_decay = None, is_training = None, name=None):
    input_dims = x.get_shape()[-1].value
    kernel_shape = kernel_size + [input_dims, output_dims]
    # kernel = tf.Variable(
    #     tf.glorot_uniform_initializer()(shape = kernel_shape),
    #     dtype = tf.float32, trainable = True, name = _kernel)

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        kernel = tf.get_variable(initializer=tf.glorot_normal_initializer()(shape = kernel_shape), dtype=tf.float32,trainable=True, name='kernel')
    x = tf.nn.conv2d(x, kernel, [1] + stride + [1], padding = padding)
    if use_bias:
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            bias = tf.get_variable(initializer=tf.zeros_initializer()(shape = [output_dims]), dtype=tf.float32,trainable=True, name='bias')
        x = tf.nn.bias_add(x, bias)
    if activation is not None:
        if bn:
            x = batch_norm(x, is_training = is_training, bn_decay = bn_decay, name=name)
        x = activation(x)
    return x

def batch_norm(x, is_training, bn_decay, name=None):
    input_dims = x.get_shape()[-1].value
    moment_dims = list(range(len(x.get_shape()) - 1))
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        beta = tf.get_variable(initializer=tf.zeros_initializer()(shape = [input_dims]), dtype=tf.float32,trainable=True, name='beta')
        gamma = tf.get_variable(initializer=tf.ones_initializer()(shape = [input_dims]), dtype=tf.float32,trainable=True, name='gamma')
    batch_mean, batch_var = tf.nn.moments(x, moment_dims, name='moments')

    decay = bn_decay if bn_decay is not None else 0.9
    ema = tf.train.ExponentialMovingAverage(decay = decay)
    # Operator that maintains moving averages of variables.
    ema_apply_op = tf.cond(
        is_training,
        lambda: ema.apply([batch_mean, batch_var]),
        lambda: tf.no_op())
    # Update moving average and return current batch's avg and var.
    def mean_var_with_update():
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)
    # ema.average returns the Variable holding the average of var.
    mean, var = tf.cond(
        is_training,
        mean_var_with_update,
        lambda: (ema.average(batch_mean), ema.average(batch_var)))
    x = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return x

def dropout(x, drop, is_training):
    x = tf.cond(
        is_training,
        lambda: tf.nn.dropout(x, rate = drop),
        lambda: x)
    return x