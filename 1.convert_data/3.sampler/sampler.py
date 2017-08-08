#
# File: sampler.py
# Date:05.04.2017
# Author: Denis Tananaev
#
#
#include libs
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import random

from six.moves import xrange
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import sys
import skimage.io as io
import scipy.misc
from numpy import newaxis
import matplotlib.pyplot as plt

#datasets
sequence_len=25 #maximum 25 
BATCH_SIZE = 1

import argparse

try:
     parser = argparse.ArgumentParser()
     parser.add_argument("sample_number", help="display a decode pair of image and depth from tf records defined by this number ",
                type=int)
     args = parser.parse_args()

except:
     e = sys.exc_info()[0]
     #print e
     

#get the list of dataset
def parse_tfrecords_folder(path):
  list_data=[] 
  #create list of images
  for file in os.listdir(path):
    if file.endswith(".tfrecords"):
      list_data.append(path+file)
      #print(path+file)
  return list_data


def parse_example_proto(example_serialized,counter):
    
    extrName='data/'+str(counter)+'/extrinsics'
    relName='data/'+str(counter)+'/relative_tf'
    imJpegName='data/'+str(counter)+'/image_jpg'
    depthPngName='data/'+str(counter)+'/depth_png'
    
    result = tf.parse_single_example(example_serialized, 
      features={
          extrName: tf.FixedLenFeature([1], tf.string),
          relName: tf.FixedLenFeature([1], tf.string),
          imJpegName: tf.FixedLenFeature([1], tf.string),
          depthPngName: tf.FixedLenFeature([1], tf.string)
          }) 
    im = tf.reshape(result[imJpegName], shape=[])
    dp = tf.reshape(result[depthPngName] , shape=[])
    relatetf=tf.decode_raw(result[relName],tf.float64)
    im=tf.image.decode_jpeg(im)
    dp=tf.image.decode_png(dp,dtype=tf.uint16)
    return im, dp,relatetf


def read_and_decode(filename_queue):
    
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    image_seq, depth_seq, transf_seq = [], [], []
    #get the size of sequence
    
    for i in range(sequence_len):
        
        im , dp, relatetf =parse_example_proto(serialized_example,i) 


        im = tf.cast(im, tf.float32)
        dp = tf.cast(dp, tf.float32)
        im.set_shape([192, 256, 3])
        dp.set_shape([192, 256, 1])  
        
        im = tf.reshape(im, [1, 192, 256, 3])
        dp = tf.reshape(dp, [1, 192, 256, 1])
        relatetf=tf.reshape(relatetf,[1,12])
        
        transf_seq.append(relatetf)        
        image_seq.append(im)
        depth_seq.append(dp)

        
    transf_seq =tf.concat(transf_seq,axis=0)
    image_seq = tf.concat(image_seq,axis=0)
    depth_seq = tf.concat(depth_seq, axis=0)
    
    image_batch,depth_batch, transf_batch= tf.train.batch(
      [image_seq,depth_seq, transf_seq],
      BATCH_SIZE,
      num_threads=1,
      capacity=1)
      
    return image_batch, depth_batch, transf_batch

import moviepy.editor as mpy
def npy_to_gif(npy, filename):
    clip = mpy.ImageSequenceClip(list(npy), fps=10)
    clip.write_gif(filename)
    
def sampler():
    input_list=[]
    input_list.append(sys.argv[1])
    sess = tf.InteractiveSession()
    filename_queue =tf.train.string_input_producer(input_list,shuffle=True)   
    image,depth,transf=read_and_decode(filename_queue)
    tf.train.start_queue_runners(sess)
    sess.run(tf.global_variables_initializer())
    im, dp, tfrans = sess.run([image,depth,transf])
    print(dp.shape)
    print("min ",np.amin(dp))
    print("max ",np.amax(dp))
    print(tfrans.shape)

    for i in range(BATCH_SIZE):
        video = im[i]
        depth=dp[i]
        print(depth.shape)
        npy_to_gif(video, './train_' + str(i) + '.gif')
        depth *= 255.0/depth.max() #normalize between [0, 255]
        npy_to_gif(depth, './depth_' + str(i) + '.gif')



sampler()

