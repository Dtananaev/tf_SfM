import numpy as np
from numpy import newaxis
from numpy.linalg import inv
import math

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


def depth2XYZcamera(intrinsics, depth):
    shape=depth.shape
    x, y = np.meshgrid(range(0,shape[1]),range(0,shape[0]))
    x=x[:, :,newaxis]
    y=y[:, :,newaxis] 
    XYZcamera=np.zeros((shape[0],shape[1],4))
    X=((x-intrinsics[0,2])*depth)/intrinsics[0,0]
    Y=((y-intrinsics[1,2])*depth)/intrinsics[1,1]
    XYZcamera[:,:,0]=X[:,:,0]
    XYZcamera[:,:,1]=Y[:,:,0] 
    XYZcamera[:,:,2]=depth[:,:,0]  
    XYZcamera[:,:,3]=depth[:,:,0]!=0
    return XYZcamera

def XYZcamera2depth(intrinsics, XYZcamera):
    shape=XYZcamera.shape
    depth=np.zeros((shape[0],shape[1]))
    print(depth.shape)
    X=np.zeros((shape[0],shape[1]))
    Y=np.zeros((shape[0],shape[1]))
    for x in range(0,shape[0]):
        for y in range(0,shape[1]):
            if XYZcamera[x,y,3]==1:
                Point=XYZcamera[x,y,:3]
                X[x,y]=Point[0]*intrinsics[0,0]/Point[2]+intrinsics[0,2]
                Y[x,y]=Point[1]*intrinsics[1,1]/Point[2]+intrinsics[1,2]
                xv=int(round(Y[x,y]))
                yv=int(round(X[x,y]))
                #if xv<shape[0] and xv>0 and yv<shape[1] and yv>0:
                    #depth[ xv,yv]=Point[2]
                depth[x,y]=Point[2]
                print(Point[2])

    return depth,X,Y

def camera2world(XYZcamera, extrinsics):
    shape=XYZcamera.shape
    result=np.zeros(shape)
    for x in range(0,shape[0]):
        for y in range(0,shape[1]):
            if XYZcamera[x,y,3]:
                Point=XYZcamera[x,y,:3]
                worldPoint=np.dot(extrinsics[:,:3],Point) + extrinsics[:,3]
                result[x,y,:3]=worldPoint
            result[x,y,3]=XYZcamera[x,y,3]
    return result
        
def world2camera(XYZworld, extrinsics):
    shape=XYZworld.shape
    result=np.zeros(shape)
    for x in range(0,shape[0]):
        for y in range(0,shape[1]):
            if XYZworld[x,y,3]:
                Point=XYZworld[x,y,:3]
                a=Point-extrinsics[:,3]
                worldPoint=np.dot(inv(extrinsics[:,:3]),a) 
                result[x,y,:3]=worldPoint
            result[x,y,3]=XYZworld[x,y,3]
    return result


            
            