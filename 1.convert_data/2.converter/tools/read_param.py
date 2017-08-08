import numpy as np
import skimage.io as io
import scipy.misc
import os
from numpy import newaxis
import matplotlib.pyplot as plt
from numpy.linalg import inv

#intrinsics='/misc/lmbraid11/tananaed/DepthNet/SUN3D/Dataset/brown_bm_6/brown_bm_6/intrinsics.txt'
#path_to_extrinsics='/misc/lmbraid11/tananaed/DepthNet/SUN3D/Dataset/brown_bm_6/brown_bm_6/extrinsics/'

def computeTransform(intr, extr):
    result=np.zeros(extr.shape)
    result[0,0,0]=1.0
    result[0,1,1]=1.0
    result[0,2,2]=1.0
    
    for i in range(1,extr.shape[0]):
        R1=extr[i,:,:3]
        R2=extr[i-1,:,:3]
        t1=extr[i,:,3]
        t2=extr[i-1,:,3]
        Rtemp=np.dot(inv(R2),R1)
        T=np.dot(inv(R2),t1-t2)
        result[i,:,:3]=Rtemp
        result[i,:,3]=T
        
    return result

def extract_intrinsics(intrinsics):
    with open(intrinsics,'r') as f:
        result=[]
        for l in f.readlines():
            num=l.strip().split(" ")
            num=map(float, num)
            result.append(num)
        return np.array(result)

def extract_extrinsics(path_to_extrinsics):
    list_extrinsics=os.listdir(path_to_extrinsics)
    list_extrinsics=sorted(list_extrinsics)
    string_counter=0
    temp=[]
    result=[]
    with open(path_to_extrinsics+str(list_extrinsics[-1]),'r') as f:
        for l in f.readlines():
            num=l.strip().split(" ")
            num=map(float, num)
            temp.append(num)
            string_counter+=1
            if string_counter==3:
                string_counter=0
                result.append(temp)
                temp=[]
    return np.array(result)
            






