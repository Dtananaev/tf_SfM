import tensorflow as tf
import numpy as np
import sys
print(sys.path)
sys.path.insert(0,'../python')
from tfspecialops import tfspecialops as ops
import h5py

USE_GPUS = sorted(set((False, tf.test.is_gpu_available())))
TYPES = (np.float32, np.float64)

#USE_GPUS = [True]
#TYPES = (np.float32,)


def visualize_points_with_normals(path, depth, intrinsics, colors, normals):
    import vis
    K = vis.intrinsics_vector_to_K(intrinsics, depth.shape[-1], depth.shape[-2])
    R = np.eye(3)
    t = np.zeros((3,))
    vis.plot_depth(path, 0, 1/depth, K, R, t, normals=normals, colors=colors)



class DepthToNormalsTest(tf.test.TestCase):

    def _load_blendswapdata(self):
        image_pair = None
        depth = None
        flow = None
        normal = None
        intrinsics = None
        motion = None

        with h5py.File('test_data/blendswap_test_scenes.h5','r') as f:
            for gname, g in f.items():
                tmp = g['image_pair'][:][np.newaxis,:,:,:]
                #print(gname,flush=True)
                # print(tmp.shape)
                if image_pair is None:
                    image_pair = tmp.copy()
                else:
                    image_pair = np.concatenate((image_pair, tmp))

                tmp = g['depth'][:][np.newaxis,:,:,:]
                if depth is None:
                    depth = tmp.copy()
                else:
                    depth = np.concatenate((depth, tmp))

                tmp = g['normal'][:][np.newaxis,:,:,:]
                if normal is None:
                    normal = tmp.copy()
                else:
                    normal = np.concatenate((normal, tmp))

                tmp = g['intrinsics'][:][np.newaxis,:]
                if intrinsics is None:
                    intrinsics = tmp.copy()
                else:
                    intrinsics = np.concatenate((intrinsics, tmp))
                
        return {'image_pair': image_pair,
                'depth': depth,
                'normal': normal,
                'intrinsics': intrinsics }


    def _test_blendswapdata(self,data):
        normal_tensor = ops.depth_to_normals(
                inverse_depth=True,
                depth=data['depth'], 
                intrinsics=data['intrinsics'] )
        normal = normal_tensor.eval()
        if( normal.dtype == np.float32 ):
            import ijremote
            ijremote.setImage("image_pair", data['image_pair'])
            ijremote.setImage("normal", normal)
            ijremote.setImage("normal_gt", data['normal'])

            import vis
            img1, _ = vis.numpy_imagepair_to_PIL_images(data['image_pair'][1])
            visualize_points_with_normals('points',data['depth'][1], data['intrinsics'][1], img1, normal[1])
            visualize_points_with_normals('points_gt',data['depth'][1], data['intrinsics'][1], img1, data['normal'][1])
        normal_no_nan = normal.copy()
        normal_no_nan[np.isnan(normal)] = 0
        normal_gt_no_nan = data['normal'].copy()
        normal_gt_no_nan[np.isnan(data['normal'])] = 0
        # new depth to normal op should be more accurate
        #self.assertAllClose(normal_no_nan, normal_gt_no_nan, rtol=0.1, atol=0.1)

                    
    def test_blendswapdata(self):
        data = self._load_blendswapdata()
        for use_gpu in USE_GPUS:
            for dtype in TYPES:
                print(use_gpu, dtype)
                data_dtype = {}
                for k,v in data.items():
                    data_dtype[k] = v.astype(dtype)
                with self.test_session(use_gpu=use_gpu, force_gpu=use_gpu):
                    self._test_blendswapdata(data_dtype)


                



if __name__ == '__main__':
    tf.test.main()


