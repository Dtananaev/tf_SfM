import tools.read_data_sample as rdata
import tools.read_param as rparam
import tools.transforms as transf
import numpy as np
import skimage.io as io
import scipy.misc
import tensorflow as tf
import os
from numpy import newaxis
import sys
import argparse
from numpy.linalg import inv

def computeTransform(intrinsics, path_to_extrinsics):
    intr=rparam.extract_intrinsics(intrinsics)
    extr=rparam.extract_extrinsics(path_to_extrinsics)  
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


intrinsics='/misc/lmbraid11/tananaed/DepthNet/SUN3D/Dataset/brown_bm_6/brown_bm_6/intrinsics.txt'
path_to_extrinsics='/misc/lmbraid11/tananaed/DepthNet/SUN3D/Dataset/brown_bm_6/brown_bm_6/extrinsics/'

#read camera parameters
intr=rparam.extract_intrinsics(intrinsics)
print("intrinsics", intr)

#compute relative transforms
extr=rparam.extract_extrinsics(path_to_extrinsics)
print("extrinsics",extr.shape[0])


result=computeTransform(intrinsics, path_to_extrinsics)
        
print(result[0,:,:])
print("-----------")
print(result[1,:,:])
print("-----------")    
print(result[2,:,:])
print("-----------")    
print(result.shape)   
    



