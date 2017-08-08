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
sys.path.insert(0,'../0.external/tf_specialops/python/')
from tfspecialops import tfspecialops as ops
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
        relu1=ops.leaky_relu(input=bnorm1, leak=0.1)
        #SKIP CONNECTION 0
        
        #Layer 2 - Downsample:Output Size 96x128x64
        conv2=cnv.conv(relu1,'conv2',[9, 9, 32, 64],stride=[1,2,2, 1],padding='SAME',wd=WEIGHT_DECAY,FLOAT16=FLOAT16)
        bnorm2=bn.batch_norm_layer(conv2,train_phase=phase_train,scope_bn='BN2')
        relu2=ops.leaky_relu(input=bnorm2, leak=0.1)
        
        #Layer 3:Output Size 96x128x64  
        conv3=cnv.conv(relu2,'conv3',[3, 3, 64, 64],wd=WEIGHT_DECAY,FLOAT16=FLOAT16) 
        bnorm3=bn.batch_norm_layer(conv3,train_phase=phase_train,scope_bn='BN3')
        relu3=ops.leaky_relu(input=bnorm3, leak=0.1)
        #SKIP CONNECTION 1
        
        #Layer 4 - Downsample:Output Size 48x64x128 
        conv4=cnv.conv(relu3,'conv4',[7, 7, 64, 128],stride=[1,2,2, 1],padding='SAME',wd=WEIGHT_DECAY,FLOAT16=FLOAT16)
        bnorm4=bn.batch_norm_layer(conv4,train_phase=phase_train,scope_bn='BN4')
        relu4=ops.leaky_relu(input=bnorm4, leak=0.1)   
        
        #Layer 5:Output Size 48x64x128 
        conv5=cnv.conv(relu4,'conv5',[3, 3, 128, 128],wd=WEIGHT_DECAY,FLOAT16=FLOAT16)
        bnorm5=bn.batch_norm_layer(conv5,train_phase=phase_train,scope_bn='BN5')
        relu5=ops.leaky_relu(input=bnorm5, leak=0.1)
        #SKIP CONNECTION 2    
        
        #Layer 6 Downsample:Output Size 24x32x256
        conv6_1=cnv.conv(relu5,'conv6_1',[5, 1, 128, 256],stride=[1,2,1, 1],padding='SAME',wd=WEIGHT_DECAY,FLOAT16=FLOAT16)
        conv6_2=cnv.conv(conv6_1,'conv6_2',[1, 5, 256, 256],stride=[1,1,2, 1],padding='SAME',wd=WEIGHT_DECAY,FLOAT16=FLOAT16)
        bnorm6=bn.batch_norm_layer(conv6_2,train_phase=phase_train,scope_bn='BN6')
        relu6=ops.leaky_relu(input=bnorm6, leak=0.1)
        
        #Layer 7:Output Size 24x32x256 
        conv7_1=cnv.conv(relu6,'conv7_1',[3, 1, 256, 256],wd=WEIGHT_DECAY,FLOAT16=FLOAT16) 
        conv7_2=cnv.conv(conv7_1,'conv7_2',[1, 3, 256, 256],wd=WEIGHT_DECAY,FLOAT16=FLOAT16)
        bnorm7=bn.batch_norm_layer(conv7_2,train_phase=phase_train,scope_bn='BN7')
        relu7=ops.leaky_relu(input=bnorm7, leak=0.1)
        #SKIP CONNECTION 3  
        
        #Layer 8 Downsample:Output Size 12x16x512 
        conv8_1=cnv.conv(relu7,'conv8_1',[3, 1, 256, 512],stride=[1,2,1, 1],padding='SAME',wd=WEIGHT_DECAY,FLOAT16=FLOAT16) 
        conv8_2=cnv.conv(conv8_1,'conv8_2',[1, 3, 512, 512],stride=[1,1,2, 1],padding='SAME',wd=WEIGHT_DECAY,FLOAT16=FLOAT16) 
        bnorm8=bn.batch_norm_layer(conv8_2,train_phase=phase_train,scope_bn='BN8')
        relu8=ops.leaky_relu(input=bnorm8, leak=0.1)
        
        #Layer 9:Output Size 12x16x512
        conv9_1=cnv.conv(relu8,'conv9_1',[1, 3, 512, 512],wd=WEIGHT_DECAY,FLOAT16=FLOAT16)
        conv9_2=cnv.conv(conv9_1,'conv9_2',[3, 1, 512, 512],wd=WEIGHT_DECAY,FLOAT16=FLOAT16)
        bnorm9=bn.batch_norm_layer(conv9_2,train_phase=phase_train,scope_bn='BN9')
        relu9=ops.leaky_relu(input=bnorm9, leak=0.1)
                
        #GO UP            
        #Layer 10 UP 1:Output Size 24x32x256
        conv10=dcnv.deconv(relu9,[BATCH_SIZE,int(IMAGE_SIZE_H/8),int(IMAGE_SIZE_W/8),256],'deconv1',[4, 4, 256, 512],stride=[1, 2, 2, 1],padding='SAME',wd=WEIGHT_DECAY,FLOAT16=FLOAT16)
        bnorm10=bn.batch_norm_layer(conv10,train_phase=phase_train,scope_bn='BN10')
        relu10=ops.leaky_relu(input=bnorm10, leak=0.1)
                
        #Layer 11 UP 1:Output 24x32x256        
        conv11=cnv.conv(relu10+relu7,'conv11',[3,3, 256, 256],wd=WEIGHT_DECAY,FLOAT16=FLOAT16)
        bnorm11=bn.batch_norm_layer(conv11,train_phase=phase_train,scope_bn='BN11')
        relu11=ops.leaky_relu(input=bnorm11, leak=0.1)       
        
        #Layer 12 UP 2:Output Size 48x64x128  
        conv12=dcnv.deconv(relu11,[BATCH_SIZE,int(IMAGE_SIZE_H/4),int(IMAGE_SIZE_W/4),128],'deconv2',[4, 4, 128, 256],stride=[1, 2, 2, 1],padding='SAME',wd=WEIGHT_DECAY,FLOAT16=FLOAT16)
        bnorm12=bn.batch_norm_layer(conv12,train_phase=phase_train,scope_bn='BN12')
        relu12=ops.leaky_relu(input=bnorm12, leak=0.1)  
                
        #Layer 13 UP 2:Output Size 48x64x128          
        conv13=cnv.conv(relu12+relu5,'conv13',[3, 3, 128, 128],wd=WEIGHT_DECAY,FLOAT16=FLOAT16)
        bnorm13=bn.batch_norm_layer(conv13,train_phase=phase_train,scope_bn='BN13')
        relu13=ops.leaky_relu(input=bnorm13, leak=0.1) 
        
        #Layer 14 UP 3:Output Size 96x128x64    
        conv14=dcnv.deconv(relu13,[BATCH_SIZE,int(IMAGE_SIZE_H/2),int(IMAGE_SIZE_W/2),64],'deconv3',[4, 4, 64, 128],stride=[1, 2, 2, 1],padding='SAME',wd=WEIGHT_DECAY,FLOAT16=FLOAT16)
        bnorm14=bn.batch_norm_layer(conv14,train_phase=phase_train,scope_bn='BN14')
        relu14=ops.leaky_relu(input=bnorm14, leak=0.1) 
        
        #Layer 15 UP 3:Output Size 96x128x64  
        conv15=cnv.conv(relu14+relu3,'conv15',[3, 3, 64, 64],wd=WEIGHT_DECAY,FLOAT16=FLOAT16)
        bnorm15=bn.batch_norm_layer(conv15,train_phase=phase_train,scope_bn='BN15')
        relu15=ops.leaky_relu(input=bnorm15, leak=0.1) 
        
        #Layer 16 UP 4:Output Size 192x256x32      
        conv16=dcnv.deconv(relu15,[BATCH_SIZE,int(IMAGE_SIZE_H),int(IMAGE_SIZE_W),32],'deconv4',[4, 4, 32, 64],stride=[1, 2, 2, 1],padding='SAME',wd=WEIGHT_DECAY,FLOAT16=FLOAT16)
        bnorm16=bn.batch_norm_layer(conv16,train_phase=phase_train,scope_bn='BN16')
        relu16=ops.leaky_relu(input=bnorm16, leak=0.1) 
        
        #Layer 17:Output Size 192x256x32   
        conv17=cnv.conv(relu16+relu1,'conv17',[3, 3, 32, 32],wd=WEIGHT_DECAY,FLOAT16=FLOAT16) 
        bnorm17=bn.batch_norm_layer(conv17,train_phase=phase_train,scope_bn='BN17')
        relu17=ops.leaky_relu(input=bnorm17, leak=0.1)  
        
        #Layer 18:Output Size 192x256x2 - 2 depth images  
        depth=cnv.conv(relu17,'scores',[3, 3, 32, 1],wd=0,FLOAT16=FLOAT16)

        
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

