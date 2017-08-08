#
# File: cnn_data.py
# Date:25.01.2017
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


sys.path.insert(0, '../0.layers/')

from six.moves import xrange
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import param
import losses as lss
#datasets
train_folder=param.tfrecords_train_folder
test_folder=param.tfrecords_test_folder

#parameters of the datasets
IMAGE_SIZE_W=param.IMAGE_SIZE_W
IMAGE_SIZE_H=param.IMAGE_SIZE_H
FLOAT16=param.FLOAT16

#parameters of the batch
sequence_len=param.SEQUENCE_LEN
BATCH_SIZE=param.BATCH_SIZE
LEARNING_RATE_DECAY_FACTOR=param.LEARNING_RATE_DECAY_FACTOR

#parameters of the data uploading
NUM_PREPROCESS_THREADS=param.NUM_PREPROCESS_THREADS
NUM_READERS=param.NUM_READERS
INPUT_QUEUE_MEMORY_FACTOR=param.INPUT_QUEUE_MEMORY_FACTOR


def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]
#get the list of dataset
def parse_tfrecords_folder(path):
  list_data=[] 
  res=get_immediate_subdirectories(path)
  print(res)
  #create list of images
  for i in range(0,len(res)):
    for file in os.listdir(path+res[i]):
        if file.endswith(".tfrecords"):
            list_data.append(path+res[i]+"/"+file)
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
        im , dp, rtf =parse_example_proto(serialized_example,i) 
        im = tf.cast(im, tf.float32)
        dp = tf.cast(dp, tf.float32)
        
        im.set_shape([192, 256, 3])
        dp.set_shape([192, 256, 1])  
        
        #im = tf.reshape(im, [1, 192, 256, 3])
        #dp = tf.reshape(dp, [1, 192, 256, 1])
        rtf=tf.cast(rtf,tf.float32)
        rtf=tf.reshape(rtf,[12])
        #preprocessing
        im=tf.image.per_image_standardization(im)
        
        transf_seq.append(rtf)
        image_seq.append(im)
        depth_seq.append(dp)

        
    transf_seq =tf.concat(transf_seq,axis=0)#concat in 1 vector 
    image_seq = tf.concat(image_seq,axis=2)#concat via channel dimension 
    depth_seq = tf.concat(depth_seq, axis=2)#concat via channel dimension
        
    return image_seq, depth_seq, transf_seq


def batch_inputs(eval_data,batch_size=BATCH_SIZE, num_preprocess_threads=NUM_PREPROCESS_THREADS,num_readers=NUM_READERS):

    with tf.name_scope('batch_processing'):
        #1 checks the parameters of the threads
        if num_preprocess_threads < 1:
            raise ValueError('Please make num_preprocess_threads at least 1')
        if num_readers < 1:
            raise ValueError('Please make num_readers 1') 

        #2 allocate the size of the queue 
        #Size the random shuffle queue to balance between good global
        # mixing (more examples) and memory use (fewer examples).
        # 1 image uses 256*192*3*4 bytes + 1 depth 256*192*1*4 bytes  = 1MB
        # The default input_queue_memory_factor is 4 implying a shuffling queue
        # size: examples_per_shard * 4 * 1MB = 4GB
        examples_per_shard = 1024
        min_queue_examples = examples_per_shard * INPUT_QUEUE_MEMORY_FACTOR
        print ('Filling queue with %d data samples before start. This will take a few minutes.' % min_queue_examples)

        #3 choose train or test set
        if eval_data==False:
            files=train_folder
        else:
            files=test_folder
            
        #4 read data 
        #read the list of pathes to tf records
        input_list=parse_tfrecords_folder(files)
        #create the list of data 
        filename_queue =tf.train.string_input_producer(input_list,shuffle=True)
        
        image,depth,transform =read_and_decode(filename_queue)
        
        
        #5 create batches
        im, dp,trf = tf.train.batch([image, depth,transform],batch_size,
        num_preprocess_threads, capacity=min_queue_examples + 3 * batch_size)
        
        #6 set depth to meters
        dp =tf.scalar_mul(0.001,dp)  
        #7 inverse depth
        dp = lss.inverse(dp) 
        
    return im, dp, trf


def read_dataset(eval_data=False):
  # Force all input processing onto CPU in order to reserve the GPU for
  # the forward inference and back-propagation
  with tf.device('/cpu:0'):
    images, depths, transforms = batch_inputs(eval_data)
    
    return images, depths,transforms







