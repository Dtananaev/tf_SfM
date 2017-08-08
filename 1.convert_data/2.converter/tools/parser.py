#
# Author: Denis Tananaev
# File: parser.py
# Date:9.02.2017
# Description: parser tool for the files for SUN3D dataset
#

#include libs
import numpy as np
import glob
import os, sys
import re
import tensorflow as tf


def read_pathes(Path_to_dataset_folder,list_trainset):
     
  with open(list_trainset) as f:          
    content = f.read().splitlines() 
    for item in range(0,len(content)):
        content[item]=Path_to_dataset_folder+content[item] 
        
    return content



def sync(path):
  images=[] 
  image={}
  r = re.compile('[ -.]+')
  #create list of images
  for file in os.listdir(path+"/image/"):
    if file.endswith(".jpg"):
      temp=r.split(file)
      image[int(temp[1])]=file  #create map of files names and timestamps
      images.append(int(temp[1])) # create arrray of timestamps
      
  images=sorted(images)
  depths=[] 
  depth={}
  #create list of images
  for file in os.listdir(path+"/depth/"):
    if file.endswith(".png"):
      temp=r.split(file)
      depth[int(temp[1])]=file  #create map of files names and timestamps
      depths.append(int(temp[1])) # create arrray of timestamps
  depths=sorted(depths)
  depths=np.array(depths)

  counter=0
  result_images=[]
  result_depths=[]
  name_depths=[]
  for i in range(0,len(images)):
    result=abs(depths-images[i])
    result=np.array(result)
    ind=np.argmin(result)
    result_depths.append(depth[depths[ind]])
    result_images.append(image[images[i]])

  return result_images, result_depths  


def sync_image2depth(path_list):
  images=[]
  depths=[]
  for path in range(0,len(path_list)):  
    im,dpth=sync(path_list[path])
    depths.extend(dpth)
    images.extend(im)
    
  return images,depths
        
        
        
def write_txtfile(data,path): 
    
   '''Write array of strings to the file specified in path'''
   with open(path, 'w') as f:
    for item in data:
        f.write("%s\n" % item)

def createPairs(input_list,output_list):
    '''Create array of tuples (image,depth) '''
    result=[]
    for i in range(0,len(input_list)):
        temp=(input_list[i],output_list[i])
        result.append(temp)
    return result
       
    
def makeLists(Path_to_dataset_folder, sequence):
    '''Make list of images samples and depth samples in two separate filse with pathes'''
    TR_path=Path_to_dataset_folder+'/'+sequence
    TR_images,TR_depth=sync(TR_path)
    dataPairs=createPairs(TR_images,TR_depth)
    
    return dataPairs
    
    #write_txtfile(TR_depth,output_tr_depth)
    #write_txtfile(TR_images,output_tr_image)
    #write_txtfile(Tst_depth,output_test_depth)
    #write_txtfile(Tst_images,output_test_image)



