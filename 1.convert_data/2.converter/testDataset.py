import numpy as np
import skimage.io as io
import scipy.misc
import tensorflow as tf
import os
from numpy import newaxis
import matplotlib.pyplot as plt
import sys
import argparse
from numpy.linalg import inv
sys.path.insert(0,'/misc/lmbraid11/tananaed/Thesis/me/1.Data_layer/tools/')
import read_param
import parser
import re
FLAGS = tf.app.flags.FLAGS

#tf.app.flags.DEFINE_string('dataset_dir', 
#                           '/misc/lmbraid12/datasets/public/sun3d/data',
#                           """Path to sun3d dataset """)

#tf.app.flags.DEFINE_string('tfrecords_dir_train', './dataset/train/',
#                          """Directory where to save tfrecords """)

#tf.app.flags.DEFINE_string('tfrecords_dir_test', './dataset/test/',
#                           """Directory where to save tfrecords """)


def read_file(textfile):
  '''Read txt file and output array of strings line by line '''
  with open(textfile) as f:
    result = f.read().splitlines()
  return result


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


class paramdecoder(object):
    
  def __init__(self):
    # Create a single Session to run all image coding calls.
    self._sess = tf.Session()
#DECODERS
    # Initializes function that decode extrinsics .
    self._decode_param_data = tf.placeholder(dtype=tf.string)
    self._decode_param = tf.decode_raw(self._decode_param_data, tf.float64)
    
  def decode_param(self, param_data):
    param=self._sess.run(self._decode_param,
                          feed_dict={self._decode_param_data: param_data})
    return param

def load_image(filename, coder):
    # Read the image file.
    with tf.gfile.FastGFile(filename, 'r') as f:
        image_data = f.read()
    image=coder.decode_jpeg(image_data) 
    return image 
      
def load_depth(filename, coder):
    # Read the image file.
    with tf.gfile.FastGFile(filename, 'r') as f:
        image_data = f.read()
    depth=coder.decode_png(image_data) 
    depth=np.right_shift(depth,3) # shift 3 bits in order to obtain right depth values
    return depth

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
  

#get the list of dataset
def parse_tfrecords_folder(path):
  list_data=[]
  number=[]
  r = re.compile('_')
  #create list of images
  for file in os.listdir(path):
    if file.endswith(".tfrecords"):
      temp=r.split(file)
      list_data.append(path+file)
      number.append(int(temp[0]))
      #print(path+file)
  return number,list_data

def load_image(filename, coder):
    # Read the image file.
    with tf.gfile.FastGFile(filename, 'r') as f:
        image_data = f.read()
    image=coder.decode_jpeg(image_data) 
    return image 
      
def load_depth(filename, coder):
    # Read the image file.
    with tf.gfile.FastGFile(filename, 'r') as f:
        image_data = f.read()
    depth=coder.decode_png(image_data) 
    depth=np.right_shift(depth,3) # shift 3 bits in order to obtain right depth values
    return depth

def check_sequence(path_to_dataset,tfrecords_sequence, sequence,pair):
    failed_data=[]
    
    print(tfrecords_sequence)
    print(sequence)
    coder = ImageCoder()
    rparam= paramdecoder()
    resizer= Resizer()
    #make list of tf records
    number, tf_list=parse_tfrecords_folder(tfrecords_sequence)
    intrinsics=path_to_dataset+"/"+sequence+"/"+ "intrinsics.txt"
    path_to_extrinsics=path_to_dataset+"/"+sequence+"/"+ "extrinsics/"
    #load extrinsics
    GTintr=read_param.extract_intrinsics(intrinsics)
    GTextr=read_param.extract_extrinsics(path_to_extrinsics)
    GTrtransf=read_param.computeTransform(GTintr,GTextr)
    for i in range(0, len(tf_list)):
        print("tfrecords ",tf_list[i])
        ind=number[i]-25
        record_iterator = tf.python_io.tf_record_iterator(path=tf_list[i])
        
        for string_record in record_iterator:
            example = tf.train.Example()
            example.ParseFromString(string_record)
            for j in range(0,25):
                extrName='data/'+str(j)+'/extrinsics'
                relName='data/'+str(j)+'/relative_tf'
                imJpegName='data/'+str(j)+'/image_jpg'
                depthPngName='data/'+str(j)+'/depth_png'
            
                extrinsics_tf=(example.features.feature[extrName]
                                  .bytes_list
                                  .value[0])
                reltf_tf=(example.features.feature[relName]
                                  .bytes_list
                                  .value[0])
                img_tf= (example.features.feature[imJpegName]
                                  .bytes_list
                                  .value[0])
                depth_tf= (example.features.feature[depthPngName]
                                  .bytes_list
                                  .value[0])
            
                reconstructed_img = coder.decode_jpeg(img_tf)
                reconstructed_depth = coder.decode_png(depth_tf)
                extr=rparam.decode_param(extrinsics_tf)
                rtransf=rparam.decode_param(reltf_tf)
                extr=np.reshape(extr,(3,4))    
                rtransf=np.reshape(rtransf,(3,4))  
                #print("min d ",np.amin(reconstructed_depth))
                #print("max d ",np.amax(reconstructed_depth))
                #ground truth
                img,depth=pair[ind+j]
                gtextr= GTextr[ind+j]
                gtrtransf=GTrtransf[ind+j]
                #check relative transforms
                if(j==0):
                    rtransf[0,0]=rtransf[0,0]-1
                    rtransf[1,1]=rtransf[1,1]-1
                    rtransf[2,2]=rtransf[2,2]-1
                    diff_rt=sum(sum(rtransf))
                else:
                    diff_rt=sum(sum(abs(rtransf-gtrtransf)))
                    
                #check extrinsics
                diff_extr=sum(sum(abs(extr-gtextr)))
                extr
                #load gt images 
                img_path=path_to_dataset+"/"+sequence+"/image/"+img
                depth_path=path_to_dataset+"/"+sequence+"/depth/"+depth
                imagejpg=load_image(img_path,coder)
                depthpng=load_depth(depth_path,coder)
                #resize        
                imagejpg=imagejpg[newaxis,:, :,:]
                imagejpg=resizer.resize_jpeg(imagejpg,[192,256])
                depthpng=depthpng[newaxis,:, :,:]
                depthpng=resizer.resize_png(depthpng,[192,256])
                
                #check 
                diff=abs(imagejpg[0,:,:,:]-reconstructed_img)/(192*256*3)
                im_check=sum(sum(sum(diff)))
                d_diff=abs(depthpng[0,:,:,:]-reconstructed_depth)
                depth_check=sum(sum(sum(d_diff)))
                
                if(im_check>5):#difference in jpeg intensity values +/- 5 
                    print("JPEG Test Failed")
                    failed_data.append(tf_list[i]+" "+"JPEG Test Failed")
                    break

                if(depth_check!=0):
                    print("PNG Test Failed")
                    failed_data.append(tf_list[i]+" "+"PNG Test Failed")
                    break
                if(diff_rt!=0):
                    print("Relative tranform Test Failed")
                    failed_data.append(tf_list[i]+" "+"Relative tranform Test Failed")
                    break                    
                if(diff_extr!=0):
                    print("Extrinsics  Test Failed")
                    failed_data.append(tf_list[i]+" "+"Extrinsics  Test Failed")
                    break 
                
                
    return failed_data
                #assert(im_check<=5)#difference in jpeg intensity values +/- 5 
                #assert(depth_check==0)
                #assert(diff_rt==0)
                #assert(diff_extr==0)

def test():    
    list_train=read_file('./train_list.txt')  
    for i in range(0,len(list_train)):
        data_folder=FLAGS.tfrecords_dir_train+"sequence_"+str(i)+"/"
        print(data_folder)
        print("Testing the sequence: ", list_train[i])
        pair=parser.makeLists(FLAGS.dataset_dir,list_train[i])    
        check_sequence(FLAGS.dataset_dir,data_folder, list_train[i],pair)

#test()