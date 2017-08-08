import numpy as np
import skimage.io as io
import scipy.misc
import tensorflow as tf
import os
from numpy import newaxis
import matplotlib.pyplot as plt
import sys
import argparse


#tfrecords_filename='../dataset/train_sun3d.tfrecords'

class ImageCoder(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Create a single Session to run all image coding calls.
    self._sess = tf.Session()
#ENCODERS
    # Initializes function that encode PNG .
    self._encode_png_data = tf.placeholder(dtype=tf.uint16)
    self._encode_png = tf.image.encode_png(self._encode_png_data)
     
    # Initializes function that encode RGB JPEG data.
    self._encode_jpeg_data = tf.placeholder(dtype=tf.uint8)
    self._encode_jpeg = tf.image.encode_jpeg(self._encode_jpeg_data)
#DECODERS
    # Initializes function that decode PNG .
    self._decode_png_data = tf.placeholder(dtype=tf.string)
    self._decode_png = tf.image.decode_png(self._decode_png_data,dtype=tf.uint16)
    
    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data)    
    
  def encode_png(self, image_data):
    image=self._sess.run(self._encode_png,
                          feed_dict={self._encode_png_data: image_data})
    return image


  def encode_jpeg(self, image_data):
    image = self._sess.run(self._encode_jpeg,
                           feed_dict={self._encode_jpeg_data: image_data})
    return image    
    
  def decode_png(self, image_data):
    image=self._sess.run(self._decode_png,
                          feed_dict={self._decode_png_data: image_data})
    return image


  def decode_jpeg(self, image_data):
    image = self._sess.run(self._decode_jpeg,
                           feed_dict={self._decode_jpeg_data: image_data})
    return image


def read_data(tfrecords_filename, sample_number):
  counter=0

  record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)
  coder = ImageCoder()


  for string_record in record_iterator:

    counter+=1
    image=None
    Depth=None	
    if counter==int(sample_number):
        print('example number:',counter)
        example = tf.train.Example()
        example.ParseFromString(string_record)
    
        height = int(example.features.feature['height']
                                 .int64_list
                                 .value[0])
    
        width = int(example.features.feature['width']
                                .int64_list
                                .value[0])
    
        img_string = (example.features.feature['image_jpg']
                                  .bytes_list
                                  .value[0])
    
        depth_string = (example.features.feature['depth_png']
                                .bytes_list
                                .value[0])
    
        Image=coder.decode_jpeg(img_string)
        Depth=coder.decode_png(depth_string)
        break

  return Image, Depth		

