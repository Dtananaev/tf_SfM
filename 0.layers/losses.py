#
# Author: Denis Tananaev
# File: losses.py
# Date: 9.02.2017
# Description: loss functions for neural networks
#
#include libs
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function




import os
import re
import sys
import tarfile  
import math 
import tensorflow as tf 

#Benjamin modules
sys.path.insert(0,'../0.external/tf_specialops/python/')
from tfspecialops import tfspecialops as ops

LOSSES_COLLECTION = 'losses'

def inverse(depth):
    inverse=tf.divide(tf.ones_like(depth),depth)
    inverse=tf.where(tf.is_nan(inverse),tf.zeros_like(inverse),inverse)
    inverse=tf.where(tf.is_inf(inverse),tf.zeros_like(inverse),inverse)
    return inverse

def convertNHWC2NCHW(data):
       #[N,H,W,C] = [N,C,H,W]
        out = tf.transpose(data, [0, 3, 1, 2])
        return out
    
def convertNCHW2NHWC(data):
        #[N,C,H,W]=[N,H,W,C] 
        out = tf.transpose(data, [0, 2, 3, 1])
        return out
    
def scinv_gradloss(output, gt,scope=None,weight=1.0):    
    with tf.name_scope(scope, 'scinv_gradloss', [output, gt]):
        # convert from NHWC to NCHW
        output=convertNHWC2NCHW(output)
        gt=convertNHWC2NCHW(gt) 
        
        # compute mask and make zero areas NaN in order to remove them later
        zero=tf.zeros_like(gt)
        mask = tf.not_equal(gt, zero)
        mask=tf.cast(mask,tf.float32)
        n=tf.reduce_sum(mask)# number of elements for evaluation
        gt_nan = tf.divide(gt,mask) # make NaN the areas where no need to eval
        epsilon=0.000000001
        # compute scale invariant grad loss                
        grad_output = ops.scale_invariant_gradient(input=output, deltas=[1,2,4,8,16], weights=[1,1,1,1,1], epsilon=epsilon)
        grad_gt = ops.scale_invariant_gradient(input=gt_nan, deltas=[1,2,4,8,16], weights=[1,1,1,1,1], epsilon=epsilon)
                        
        diff = ops.replace_nonfinite(grad_gt-grad_output)
    
        gradLoss=tf.reduce_sum(tf.sqrt(tf.reduce_sum(diff**2, axis=1)+epsilon))/n
        tf.add_to_collection('losses', weight*gradLoss)



def L1loss_transforms(result,gt,scope=None,weight=1.0):
    with tf.name_scope(scope, 'L1loss_transforms', [result, gt]):
        
        diff=tf.reduce_mean(tf.abs(tf.subtract(result,gt)))
        tf.add_to_collection('losses', weight*diff)
        
        
def L1loss_depth(result,gt,scope=None,weight=1.0):
    with tf.name_scope(scope, 'L1loss', [result, gt]):
        #don't eval on zero points in depth images
        zero=tf.zeros_like(gt)
        mask = tf.not_equal(gt, zero)
        mask=tf.cast(mask,tf.float32)
        n=tf.reduce_sum(mask)# number of elements for evaluation
        diff = tf.subtract(result,gt)
        res=tf.reduce_sum(abs(tf.multiply(mask,diff)))/n
        
        tf.add_to_collection('losses', weight*res)
        
        
def L2loss_depth(output, gt,scope=None,weight=1.0):
    with tf.name_scope(scope, 'L2loss_depth', [output, gt]):
        zero=tf.zeros_like(gt)
        mask = tf.not_equal(gt, zero)
        mask=tf.cast(mask,tf.float32)
        n=tf.reduce_sum(mask)# number of elements for evaluation
        # compute L2 loss
        diff=tf.multiply(mask,tf.subtract(output,gt))#find difference
        L2loss=tf.reduce_sum(tf.square(diff))/n
        tf.add_to_collection('losses', weight*L2loss)



'''
def scinv_loss(result,gt,weight=1.0):
    #don't eval on zero points in depth images
    zero=tf.zeros_like(gt)
    mask = tf.not_equal(gt, zero)
    mask=tf.cast(mask,tf.float32)    
    n=tf.reduce_sum(mask)# number of    elements for evaluation

    
    d=tf.subtract(tf.log(result),tf.log(gt))
    d=tf.where(tf.is_nan(d),tf.zeros_like(d),d)
    d=tf.where(tf.is_inf(d),tf.zeros_like(d),d)
    d=tf.check_numerics(d, message='d problem', name=None)
    d=tf.multiply(mask,d)
    dsq=tf.reduce_sum(tf.square(d))
    error= (1/n)*dsq - (0.5/(n*n))* tf.square(tf.reduce_sum(d))
    tf.add_to_collection('losses', weight*error)

def L1rel_loss(result,gt,weight=1.0):
    #don't eval on zero points in depth images

    zero=tf.zeros_like(gt)
    mask = tf.not_equal(gt, zero)
    mask=tf.cast(mask,tf.float32)    
    n=tf.reduce_sum(mask)# number of elements for evaluation

    
    gt=tf.where(tf.equal(mask,zero),tf.ones_like(gt),gt)# replace all 0 by 1 in order to avoiding division by 0
    error=(1/n)*tf.reduce_sum(tf.multiply(mask,tf.abs(tf.divide(tf.subtract(result,gt),gt))))
    tf.add_to_collection('losses', weight*error)


    
def L1inv_loss(result,gt,weight=1.0):
    
    #don't eval on zero points in depth images
    zero=tf.zeros_like(gt)
    mask = tf.not_equal(gt, zero)
    mask=tf.cast(mask,tf.float32)  
    n=tf.reduce_sum(mask)
    one=tf.ones_like(gt)
    invgt=inverse(gt)
    invresult=inverse(result)
        
    
    error=tf.reduce_sum(tf.multiply(mask,tf.abs(tf.subtract(invresult,invgt))))/n
    tf.add_to_collection('losses', weight*error)

'''



        



    


