# -- coding: utf-8 --

import pandas as pd
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import csv
import datetime
import math

def uniform(shape, scale=0.05, name=None):
    """Uniform init."""
    initial = tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    # init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    # initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    # kernel = tf.Variable(initial, name=name)

    kernel = tf.Variable(
                        tf.glorot_uniform_initializer()(shape = shape),
                        dtype = tf.float32, trainable = True, name = name)
    return kernel


def zeros(shape, name=None):
    """All zeros."""
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)