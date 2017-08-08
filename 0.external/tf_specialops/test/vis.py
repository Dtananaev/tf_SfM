import pyximport; pyximport.install()
import numpy as np
import math
from PIL import Image


def intrinsics_vector_to_K(intrinsics, width, height):
    """Converts the normalized intrinsics vector to the calibration matrix K

    intrinsics: np.ndarray
        4 element vector with normalized intrinsics [fx, fy, cx, cy]

    width: int
        image width in pixels

    height: int 
        image height in pixels

    returns the calibration matrix K as numpy.ndarray
    """
    tmp = intrinsics.squeeze().astype(np.float64)
    K = np.array([tmp[0]*width, 0, tmp[2]*width, 0, tmp[1]*height, tmp[3]*height, 0, 0, 1], dtype=np.float64).reshape((3,3))
    
    return K



def compute_point_cloud_from_depthmap( depth, K, R, t, normals=None, colors=None ):
    """Creates a point cloud numpy array and optional normals and colors arrays

    depth: numpy.ndarray 
        2d array with depth values

    K: numpy.ndarray
        3x3 matrix with internal camera parameters

    R: numpy.ndarray
        3x3 rotation matrix

    t: numpy.ndarray
        3d translation vector

    normals: numpy.ndarray
        optional array with normal vectors

    colors: PIL.Image
        optional RGB image with the same dimensions as the depth map
    """
    from vis_cython import compute_point_cloud_from_depthmap as _compute_point_cloud_from_depthmap
    return _compute_point_cloud_from_depthmap(depth, K, R, t, normals, colors)




def plot_depth( path, time, depth, K, R, t, normals=None, colors=None ):
    """Plots the inverse depth map as point cloud in siasa

    path: str
        Path for the pointcloud in siasa

    time: int
        time value for the pointcloud in siasa

    depth: numpy.ndarray
        2d array with depth values

    K: numpy.ndarray
        3x3 matrix with internal camera parameters

    R: numpy.ndarray
        3x3 rotation matrix

    t: numpy.ndarray
        3d translation vector

    normals: numpy.ndarray
        optional array with normal vectors

    colors: PIL.Image
        optional RGB image with the same dimensions as the depth map

    """
    import siasainterface as siasa

    point_cloud = compute_point_cloud_from_depthmap(depth, K, R, t, normals, colors)
    
    point_attributes = []
    if not normals is None:
        point_attributes.append(siasa.AttributeArray('Normals', point_cloud['normals']))
    if not colors is None:
        point_attributes.append(siasa.AttributeArray('Colors', point_cloud['colors']))
    siasa.setPolyData(point_cloud['points'], path, timestep=time, point_attributes=point_attributes)
    return



def numpy_imagepair_to_PIL_images(imagepair):
    """Generates a PIL Image pair from the numpy array

    imagepair: numpy.ndarray
        Image pair as returned from the multivih5 data layer with range [-0.5,0.5]
    """
    tmp = imagepair.squeeze() + 0.5
    tmp *= 255
    tmp = tmp.astype(np.uint8)
    tmp1 = tmp[0:3,:,:].transpose([1,2,0])
    tmp2 = tmp[3:,:,:].transpose([1,2,0])
    return Image.fromarray(tmp1), Image.fromarray(tmp2)

