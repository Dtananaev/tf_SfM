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


def process_image(filename,file_format,coder,resizer):
    # Read the image file.
    with tf.gfile.FastGFile(filename, 'r') as f:
        image_data = f.read()
    new_size=[192,256]
    if file_format=='JPEG':
      image=coder.decode_jpeg(image_data)   
      #image=resizer.crop_jpeg(image)
      image=image[newaxis,:, :,:]
      image=resizer.resize_jpeg(image,new_size)
      height = image.shape[1]
      width = image.shape[2]
      result=coder.encode_jpeg(image[0,:,:,:])
    if file_format=='PNG':
        image=coder.decode_png(image_data) 
        image=np.right_shift(image,3) # shift 3 bits in order to obtain right depth values
       # image=resizer.crop_png(image)
        image=image[newaxis,:, :,:]
        image=resizer.resize_png(image,new_size)
        height = image.shape[1]
        width = image.shape[2]
        result=coder.encode_png(image[0,:,:,:])   
        
    return result, height, width     


def centered_crop(image,new_w,new_h):
    '''Make centered crop of the image'''
    height = image.shape[0]
    width = image.shape[1]
    left = (width - new_w)/2
    top = (height - new_h)/2
    right = (width + new_w)/2
    bottom = (height + new_h)/2
    return image[top:bottom,left:right]

def read_file(textfile):
  '''Read txt file and output array of strings line by line '''
  with open(textfile) as f:
    result = f.read().splitlines()
  return result


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def make_tfrecords(path_to_dataset, path_to_tfrecords, sequence,filename_pairs):
    '''Convert pairs of (image, depth) tuple to the tfrecords format'''
        
    name=sequence.split('/')

    
    intrinsics=path_to_dataset+"/"+sequence+"/"+ "intrinsics.txt"
    path_to_extrinsics=path_to_dataset+"/"+sequence+"/"+ "extrinsics/"
    #load extrinsics
    intr=read_param.extract_intrinsics(intrinsics)
    extr=read_param.extract_extrinsics(path_to_extrinsics)
    rtransf=read_param.computeTransform(intr,extr)
    
    coder = ImageCoder()
    resizer=Resizer()
    counter=0
    dataset_len=0
    feature={}
    intr=intr.flatten()
    feature['intrinsics']=_bytes_feature(intr.tostring())
    
    for img, depth in filename_pairs:

        img_path=path_to_dataset+"/"+sequence+"/image/"+img
        depth_path=path_to_dataset+"/"+sequence+"/depth/"+depth
        imagejpg,height, width=process_image(img_path,'JPEG',coder,resizer)
        depthpng,height, width=process_image(depth_path,'PNG',coder,resizer)        
        fExtr=extr[dataset_len,:,:].flatten()
        if(counter==0):
            fRtransf=np.zeros(12)
            fRtransf[0]=1.0
            fRtransf[5]=1.0   
            fRtransf[10]=1.0
        else:
            fRtransf=rtransf[dataset_len,:,:].flatten()
            
        imagejpg=np.array(imagejpg)
        depthpng=np.array(depthpng)
        feature['data/'+str(counter)+'/extrinsics']=_bytes_feature(fExtr.tostring())    
        feature['data/'+str(counter)+'/relative_tf']=_bytes_feature(fRtransf.tostring())  
        feature['data/'+str(counter)+'/image_jpg']=_bytes_feature(imagejpg.tostring())
        feature['data/'+str(counter)+'/depth_png']=_bytes_feature(depthpng.tostring())
        counter+=1
        dataset_len+=1
        print('files processed',dataset_len)
        if dataset_len%25==0:
            tfrecords_filename=path_to_tfrecords+str(dataset_len)+"_"+name[1]+".tfrecords"
            writer = tf.python_io.TFRecordWriter(tfrecords_filename)
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
            writer.close()
            feature={}
            feature['intrinsics']=_bytes_feature(intr.tostring())    
            counter=0



def createPairs(train_im,train_d):
    '''Create array of tuples (image,depth) '''
    #read the list of pathes to jpg data from txt
    input_list=read_file(train_im)
    #read the list of pathes to png data from txt
    output_list=read_file(train_d)
    result=[]
    for i in range(0,len(input_list)):
        temp=(input_list[i],output_list[i])
        result.append(temp)
    return result
