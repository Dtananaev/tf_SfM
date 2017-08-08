#
# Author: Denis Tananaev
# File: makeTFrecords.py
# Date:9.02.2017
# Description:  tool for the tfrecords convertion of the SUN3D dataset
#

import numpy as np
import skimage.io as io
import scipy.misc
import tensorflow as tf
import os
from numpy import newaxis
import read_param
import parser
class Resizer(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Create a single Session to run all image coding calls.
    self._sess = tf.Session()
    #CROP    
    # Initializes function that crop PNG.
    self._crop_png_data = tf.placeholder(dtype=tf.uint16)
    self._crop_png = tf.image.central_crop(self._crop_png_data,central_fraction=1.0)
    
    # Initializes function that crop JPEG.
    self._crop_jpeg_data = tf.placeholder(dtype=tf.uint8)
    self._crop_jpeg = tf.image.central_crop(self._crop_jpeg_data,central_fraction=1.0)
    #RESIZE
    self._new_size=tf.placeholder(dtype=tf.int32)
    # Initializes function that resize PNG.
    self._resize_png_data = tf.placeholder(dtype=tf.int32)
    self._resize_png = tf.image.resize_nearest_neighbor(self._resize_png_data,self._new_size)    
    # Initializes function that resize JPEG.
    self._resize_jpeg_data = tf.placeholder(dtype=tf.uint8)
    self._resize_jpeg = tf.image.resize_bilinear(self._resize_jpeg_data,self._new_size)
    
  def crop_png(self, image_data):
    image=self._sess.run(self._crop_png,
                          feed_dict={self._crop_png_data: image_data})
    return image

  def crop_jpeg(self, image_data):
    image = self._sess.run(self._crop_jpeg,
                           feed_dict={self._crop_jpeg_data: image_data})
    return image    
    
    
  def resize_png(self, image_data,new_size):
    image=self._sess.run(self._resize_png,
                          feed_dict={self._resize_png_data: image_data,
                                     self._new_size: new_size
                                     })
    return image

  def resize_jpeg(self, image_data,new_size):
    image = self._sess.run(self._resize_jpeg,
                           feed_dict={self._resize_jpeg_data: image_data,
                                      self._new_size: new_size
                                      })
    return image  

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
    self._encode_jpeg = tf.image.encode_jpeg(self._encode_jpeg_data,quality=100)
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


def process_image(filename,file_format,coder):
    # Read the image file.
    with tf.gfile.FastGFile(filename, 'r') as f:
        image_data = f.read()
    if file_format=='JPEG':
      image=coder.decode_jpeg(image_data)   

    if file_format=='PNG':
        image=coder.decode_png(image_data) 
        image=np.right_shift(image,3) # shift 3 bits in order to obtain right depth values
        
    return image


  
def load_data(path_to_dataset,sequence,filename_pairs,sample_number):

    
    coder = ImageCoder()
    img,depth=filename_pairs[sample_number]
    img_path=path_to_dataset+"/"+sequence+"/image/"+img
    depth_path=path_to_dataset+"/"+sequence+"/depth/"+depth
    imagejpg=process_image(img_path,'JPEG',coder)
    depthpng=process_image(depth_path,'PNG',coder)        

        
        
    return imagejpg,depthpng


def read_data(path_to_dataset_folder,sequence,sample_number1):
    pair=parser.makeLists(path_to_dataset_folder,sequence)
    image,depth=load_data(path_to_dataset_folder,sequence,pair,sample_number1)
    return  image,depth         


