#
# Author: Denis Tananaev
# File: activations.py
# Date: 9.02.2017
# Description: activation functions for neural networks
#

#include libs
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from six.moves import xrange

import tensorflow as tf
#import summary
import conv as cnv
import summary as sm

def ReLU(x,scope, reuse=None):
    with tf.variable_scope(scope, 'ReLU', [x], reuse=reuse):
        relu= tf.nn.relu(x)
        sm._activation_summary(relu)
        return relu  

def parametric_relu(x,scope,FLOAT16=False,reuse=None):
    with tf.variable_scope(scope, 'PReLU', [x], reuse=reuse):
        alphas =cnv._variable_on_cpu('alphas', x.get_shape()[-1], tf.truncated_normal_initializer(stddev=0.5),FLOAT16=FLOAT16)
        pos = tf.nn.relu(x)
        neg = alphas * (x - abs(x)) * 0.5
        sm._activation_summary(pos + neg)
        return pos + neg


def maxout(x,scope,FLOAT16=False,reuse=None):
    with tf.variable_scope(scope, 'MAXOUT', [x], reuse=reuse):
        shape=x.get_shape()
        w1 =cnv._variable_on_cpu('w1', x.get_shape()[-1], tf.constant_initializer(0.1),FLOAT16=FLOAT16)
        w2 =cnv._variable_on_cpu('w2', x.get_shape()[-1], tf.constant_initializer(0.1),FLOAT16=FLOAT16)
        neg = w1 * (x - abs(x)) * 0.5 
        pos = tf.nn.relu(w2*x)
        sm._activation_summary(pos + neg)
        return pos + neg

def leaky_relu(x,scope,leak=0.1,FLOAT16=False,reuse=None):
    with tf.variable_scope(scope, 'leakyReLU', [x], reuse=reuse):
        pos = tf.nn.relu(x)
        neg = leak * (x - abs(x)) * 0.5
        sm._activation_summary(pos + neg)
        return pos + neg
