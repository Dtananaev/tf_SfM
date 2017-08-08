#
# File: model.py
# Date:21.01.2017
# Author: Denis Tananaev
# 
#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
sys.path.insert(0, '../0.layers/')
import os
import re
import tarfile
import math 
import tensorflow as tf
#layers
import summary as sm
import conv as cnv
import activations as act
import deconv as dcnv
import batchnorm as bn
import losses as lss

import data
import param
#parameters of the datasets
IMAGE_SIZE_W=param.IMAGE_SIZE_W
IMAGE_SIZE_H=param.IMAGE_SIZE_H
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN=param.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = param.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
FLOAT16=param.FLOAT16
#parameters of the data uploading
BATCH_SIZE=param.BATCH_SIZE
# Constants describing the training process.
MOVING_AVERAGE_DECAY = param.MOVING_AVERAGE_DECAY   # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = param.NUM_EPOCHS_PER_DECAY     # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = param.LEARNING_RATE_DECAY_FACTOR  # Learning rate decay factor.
INITIAL_LEARNING_RATE = param.INITIAL_LEARNING_RATE     # Initial learning rate.
WEIGHT_DECAY=param.WEIGHT_DECAY


def inference(images, phase_train,scope='CNN'):
    
    with tf.name_scope(scope, [images]):
        
        #THE DEPTH NETWORK
        
        #Layer 1: Output Size 192x256x32
        conv1=cnv.conv(images,'conv1',[11, 11, 3, 32],stride=[1,1,1, 1],padding='SAME',wd=WEIGHT_DECAY,FLOAT16=FLOAT16)
        bnorm1=bn.batch_norm_layer(conv1,train_phase=phase_train,scope_bn='BN1')
        relu1=act.ReLU(bnorm1,'parametricReLU1') 
        #SKIP CONNECTION 0
        
        #Layer 2 - Downsample:Output Size 96x128x64
        conv2=cnv.conv(relu1,'conv2',[9, 9, 32, 64],stride=[1,2,2, 1],padding='SAME',wd=WEIGHT_DECAY,FLOAT16=FLOAT16)
        bnorm2=bn.batch_norm_layer(conv2,train_phase=phase_train,scope_bn='BN2')
        relu2=act.ReLU(bnorm2,'ReLU2')
        
        #Layer 3:Output Size 96x128x64  
        conv3=cnv.conv(relu2,'conv3',[3, 3, 64, 64],wd=WEIGHT_DECAY,FLOAT16=FLOAT16) 
        bnorm3=bn.batch_norm_layer(conv3,train_phase=phase_train,scope_bn='BN3')
        relu3=act.parametric_relu(bnorm3,'parametricReLU3')
        #SKIP CONNECTION 1
        
        #Layer 4 - Downsample:Output Size 48x64x128 
        conv4=cnv.conv(relu3,'conv4',[7, 7, 64, 128],stride=[1,2,2, 1],padding='SAME',wd=WEIGHT_DECAY,FLOAT16=FLOAT16)
        bnorm4=bn.batch_norm_layer(conv4,train_phase=phase_train,scope_bn='BN4')
        relu4=act.ReLU(bnorm4,'ReLU4')     
        
        #Layer 5:Output Size 48x64x128 
        conv5=cnv.conv(relu4,'conv5',[3, 3, 128, 128],wd=WEIGHT_DECAY,FLOAT16=FLOAT16)
        bnorm5=bn.batch_norm_layer(conv5,train_phase=phase_train,scope_bn='BN5')
        relu5=act.parametric_relu(bnorm5,'parametricReLU5')
        #SKIP CONNECTION 2    
        
        #Layer 6 Downsample:Output Size 24x32x256
        conv6_1=cnv.conv(relu5,'conv6_1',[5, 1, 128, 256],stride=[1,2,1, 1],padding='SAME',wd=WEIGHT_DECAY,FLOAT16=FLOAT16)
        conv6_2=cnv.conv(conv6_1,'conv6_2',[1, 5, 256, 256],stride=[1,1,2, 1],padding='SAME',wd=WEIGHT_DECAY,FLOAT16=FLOAT16)
        bnorm6=bn.batch_norm_layer(conv6_2,train_phase=phase_train,scope_bn='BN6')
        relu6=act.ReLU(bnorm6,'ReLU6')
        
        #Layer 7:Output Size 24x32x256 
        conv7_1=cnv.conv(relu6,'conv7_1',[3, 1, 256, 256],wd=WEIGHT_DECAY,FLOAT16=FLOAT16) 
        conv7_2=cnv.conv(conv7_1,'conv7_2',[1, 3, 256, 256],wd=WEIGHT_DECAY,FLOAT16=FLOAT16)
        bnorm7=bn.batch_norm_layer(conv7_2,train_phase=phase_train,scope_bn='BN7')
        relu7=act.parametric_relu(bnorm7,'parametricReLU7')
        #SKIP CONNECTION 3  
        
        #Layer 8 Downsample:Output Size 12x16x512 
        conv8_1=cnv.conv(relu7,'conv8_1',[3, 1, 256, 512],stride=[1,2,1, 1],padding='SAME',wd=WEIGHT_DECAY,FLOAT16=FLOAT16) 
        conv8_2=cnv.conv(conv8_1,'conv8_2',[1, 3, 512, 512],stride=[1,1,2, 1],padding='SAME',wd=WEIGHT_DECAY,FLOAT16=FLOAT16) 
        bnorm8=bn.batch_norm_layer(conv8_2,train_phase=phase_train,scope_bn='BN8')
        relu8=act.ReLU(bnorm8,'ReLU8') 
        
        #Layer 9:Output Size 12x16x512
        conv9_1=cnv.conv(relu8,'conv9_1',[1, 3, 512, 512],wd=WEIGHT_DECAY,FLOAT16=FLOAT16)
        conv9_2=cnv.conv(conv9_1,'conv9_2',[3, 1, 512, 512],wd=WEIGHT_DECAY,FLOAT16=FLOAT16)
        bnorm9=bn.batch_norm_layer(conv9_2,train_phase=phase_train,scope_bn='BN9')
        relu9=act.parametric_relu(bnorm9,'ReLU9')  
        
        #Layer 10: Output Size 6x8x1024
        conv10_1=cnv.conv(relu9,'conv10_1',[1, 3, 512, 1024],stride=[1,2,1, 1],wd=WEIGHT_DECAY,FLOAT16=FLOAT16)
        conv10_2=cnv.conv(conv10_1,'conv10_2',[3, 1, 1024, 1024],stride=[1,1,2, 1],wd=WEIGHT_DECAY,FLOAT16=FLOAT16)
        bnorm10=bn.batch_norm_layer(conv10_2,train_phase=phase_train,scope_bn='BN10')
        relu10=act.parametric_relu(bnorm10,'parametricReLU10')
        
        
        #GO UP 
        #Layer 11 UP 1:Output Size 12x16x512
        deconv0=dcnv.deconv(relu10,[BATCH_SIZE,int(IMAGE_SIZE_H/16),int(IMAGE_SIZE_W/16),512],'deconv0',[4, 4, 512, 1024],stride=[1, 2, 2, 1],padding='SAME',wd=WEIGHT_DECAY,FLOAT16=FLOAT16)
        dbnorm0=bn.batch_norm_layer(deconv0+relu9,train_phase=phase_train,scope_bn='dBN0')
        drelu0=act.parametric_relu(dbnorm0,'dparametricReLU0')
        
        #Layer 12 UP 1:Output Size 12x16x512        
        conv12_1=cnv.conv(drelu0,'conv12_1',[1, 3, 512, 512],wd=WEIGHT_DECAY,FLOAT16=FLOAT16)
        conv12_2=cnv.conv(conv12_1,'conv12_2',[3, 1, 512, 512],wd=WEIGHT_DECAY,FLOAT16=FLOAT16)
        bnorm12=bn.batch_norm_layer(conv12_2,train_phase=phase_train,scope_bn='BN12')
        relu12=act.parametric_relu(bnorm12,'parametricReLU12')        
        
        #Layer 13 UP 1:Output Size 24x32x256
        deconv1=dcnv.deconv(relu12,[BATCH_SIZE,int(IMAGE_SIZE_H/8),int(IMAGE_SIZE_W/8),256],'deconv1',[4, 4, 256, 512],stride=[1, 2, 2, 1],padding='SAME',wd=WEIGHT_DECAY,FLOAT16=FLOAT16)
        dbnorm1=bn.batch_norm_layer(deconv1+relu7,train_phase=phase_train,scope_bn='dBN1')
        drelu1=act.parametric_relu(dbnorm1,'dparametricReLU1')
                
        #Layer 14 UP 1:Output 24x32x256        
        conv14_1=cnv.conv(drelu1,'conv14_1',[1, 3, 256, 256],wd=WEIGHT_DECAY,FLOAT16=FLOAT16)
        conv14_2=cnv.conv(conv14_1,'conv14_2',[3, 1, 256, 256],wd=WEIGHT_DECAY,FLOAT16=FLOAT16)
        bnorm14=bn.batch_norm_layer(conv14_2,train_phase=phase_train,scope_bn='BN14')
        relu14=act.parametric_relu(bnorm14,'parametricReLU14')          
        
        #Layer 15 UP 2:Output Size 48x64x128  
        deconv2=dcnv.deconv(relu14,[BATCH_SIZE,int(IMAGE_SIZE_H/4),int(IMAGE_SIZE_W/4),128],'deconv2',[4, 4, 128, 256],stride=[1, 2, 2, 1],padding='SAME',wd=WEIGHT_DECAY,FLOAT16=FLOAT16)
        dbnorm2=bn.batch_norm_layer(deconv2+relu5,train_phase=phase_train,scope_bn='dBN2')
        drelu2=act.parametric_relu(dbnorm2,'dparametricReLU2')
        
        #Layer 16 UP 2:Output Size 48x64x128          
        conv16=cnv.conv(drelu2,'conv16',[3, 3, 128, 128],wd=WEIGHT_DECAY,FLOAT16=FLOAT16)
        bnorm16=bn.batch_norm_layer(conv16,train_phase=phase_train,scope_bn='BN16')
        relu16=act.parametric_relu(bnorm16,'parametricReLU16')
        
        #Layer 17 UP 3:Output Size 96x128x64    
        deconv3=dcnv.deconv(relu16,[BATCH_SIZE,int(IMAGE_SIZE_H/2),int(IMAGE_SIZE_W/2),64],'deconv3',[4, 4, 64, 128],stride=[1, 2, 2, 1],padding='SAME',wd=WEIGHT_DECAY,FLOAT16=FLOAT16)
        dbnorm3=bn.batch_norm_layer(deconv3+relu3,train_phase=phase_train,scope_bn='dBN3')
        drelu3=act.parametric_relu(dbnorm3,'dparametricReLU3')
        
        #Layer 18 UP 3:Output Size 96x128x64  
        conv18=cnv.conv(drelu3,'conv18',[3, 3, 64, 64],wd=WEIGHT_DECAY,FLOAT16=FLOAT16)
        bnorm18=bn.batch_norm_layer(conv18,train_phase=phase_train,scope_bn='BN18')
        relu18=act.parametric_relu(bnorm18,'parametricReLU18')
        
        #Layer 19 UP 4:Output Size 192x256x32      
        deconv4=dcnv.deconv(relu18,[BATCH_SIZE,int(IMAGE_SIZE_H),int(IMAGE_SIZE_W),32],'deconv4',[4, 4, 32, 64],stride=[1, 2, 2, 1],padding='SAME',wd=WEIGHT_DECAY,FLOAT16=FLOAT16)
        dbnorm5=bn.batch_norm_layer(deconv4+relu1,train_phase=phase_train,scope_bn='dBN4')
        drelu4=act.parametric_relu(dbnorm5,'dparametricReLU4')
        
        #Layer 20:Output Size 192x256x32   
        conv_last=cnv.conv(drelu4,'conv_last',[3, 3, 32, 32],wd=WEIGHT_DECAY,FLOAT16=FLOAT16) 
        bnorm20=bn.batch_norm_layer(conv_last,train_phase=phase_train,scope_bn='BN20')
        relu_last=act.ReLU(bnorm20,'ReLU_last')  
        #Layer 15:Output Size 192x256x2 - 2 depth images  
        depth=cnv.conv(relu_last,'scores',[3, 3, 32, 1],wd=0,FLOAT16=FLOAT16)

        
        return depth


def loss(pdepth,gtdepth):
    #L2 for 2 depth images
    lss.L1loss_depth(pdepth, gtdepth,weight=300.0)
    #lss.scinv_loss(pdepth,gtdepth,weight=1.0)
    lss.scinv_gradloss(pdepth, gtdepth,weight=1500.0)
     
    
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, name='avg')
  losses = tf.get_collection(lss.LOSSES_COLLECTION)
  loss_averages_op = loss_averages.apply(losses + [total_loss])
  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.summary.scalar(l.op.name + ' (raw)', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))

  return loss_averages_op


def train(total_loss, global_step):


  # Variables that affect learning rate.
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / BATCH_SIZE
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  tf.summary.scalar('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    #opt = tf.train.GradientDescentOptimizer(lr)
    opt=tf.train.AdamOptimizer(lr)
    grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.summary.histogram(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op

