import tools.read_data_sample as rdata
import tools.read_param as rparam
import tools.transforms as transf
import numpy as np
import skimage.io as io
import scipy.misc
import tensorflow as tf
import os
from numpy import newaxis
import matplotlib.pyplot as plt
import sys
import argparse
import tools.newreader as nr
#path to camera param
intrinsics='/misc/lmbraid11/tananaed/DepthNet/SUN3D/Dataset/brown_bm_6/brown_bm_6/intrinsics.txt'
path_to_extrinsics='/misc/lmbraid11/tananaed/DepthNet/SUN3D/Dataset/brown_bm_6/brown_bm_6/extrinsics/'

#path to tfrecords files
dataset_path='/misc/lmbraid11/tananaed/DepthNet/SUN3D/Dataset'
sequence="brown_bm_6/brown_bm_6"

sample_number1=1
sample_number2=700

#read camera parameters
intr=rparam.extract_intrinsics(intrinsics)
extr=rparam.extract_extrinsics(path_to_extrinsics)
#read depth and file 1
image1,depth1 =nr.read_data(dataset_path,sequence,sample_number1)

scipy.misc.imsave('./depth1.png', depth1[:,:,0])
scipy.misc.imsave('./image1.jpeg', image1)
#read depth and file 2
image2,depth2 =nr.read_data(dataset_path,sequence,sample_number2)
scipy.misc.imsave('./depth2.png', depth2[:,:,0])
scipy.misc.imsave('./image2.jpeg', image2)
print(depth1.shape)
print(depth2.shape)
depth1=depth1/1000.0
depth2=depth2/1000.0
#XYZ 3D points with respect to the camera position
XYZcamera=transf.depth2XYZcamera(intr, depth1)

XYZworld=transf.camera2world(XYZcamera, extr[sample_number1-1,:,:])
print('XYZworld')
print(XYZworld.shape)
projXYZCamera=transf.world2camera(XYZworld,  extr[sample_number2-1,:,:])
print('projXYZCamera')
print(projXYZCamera.shape)
#assert projXYZCamera.all()==XYZcamera.all()

projDepth,x,y=transf.XYZcamera2depth(intr, projXYZCamera)
print("of")
print("depth shape ",projDepth.shape)
print(x.shape)
print(y.shape)
scipy.misc.imsave('./projDepth.png', projDepth)



