#
# Author: Denis Tananaev
# File: ConvertDataset.py
# Date:9.02.2017
# Description: parser tool for the files for SUN3D dataset
#

#include libs
import os, sys
#sys.path.insert(0,'/misc/lmbraid11/tananaed/Thesis/me/1.Data_layer/')
import numpy as np
import glob

import re
import tensorflow as tf
import tools.parser as parser
import tools.makeTFrecords as mtf
import testDataset as tst
path_train='../folder_checker/train_list.txt'
seq=0 #start from the folder name "sequence_(seq)"

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('dataset_dir', 
                           '/misc/lmbraid12/datasets/public/sun3d/data',
                           """Path to sun3d dataset """)

tf.app.flags.DEFINE_string('tfrecords_dir_train', '../train/',
                           """Directory where to save tfrecords """)


def convert():
    
    print('dataset folder:', FLAGS.dataset_dir)    
    print('tfrecords output directory for training set:',FLAGS.tfrecords_dir_train)
    print('start parcing the dataset folders for train set')
    test_result=[]
    
    list_train=mtf.read_file(path_train)  
    pathesTrain=[]
    for i in range(0,len(list_train)):
        data_folder=FLAGS.tfrecords_dir_train+"sequence_"+str(seq+i)+"/"
        print(data_folder)
        if tf.gfile.Exists(data_folder):
           tf.gfile.DeleteRecursively(data_folder)
        tf.gfile.MakeDirs(data_folder)
        print("Converting the sequence: ", list_train[i])
        pair=parser.makeLists(FLAGS.dataset_dir,list_train[i])
        mtf.make_tfrecords(FLAGS.dataset_dir,data_folder,list_train[i] ,pair)
        #test data
        failed_data=tst.check_sequence(FLAGS.dataset_dir,data_folder, list_train[i],pair)
        test_result.extend(failed_data)
    parser.write_txtfile(test_result,"./test_fails.txt")
    
    
def main(argv=None):
  # lists_dir
  if tf.gfile.Exists(FLAGS.tfrecords_dir_train):
    tf.gfile.DeleteRecursively(FLAGS.tfrecords_dir_train)
  tf.gfile.MakeDirs(FLAGS.tfrecords_dir_train)
  convert()
  
  
if __name__ == '__main__':
  tf.app.run()  
    
    
    

