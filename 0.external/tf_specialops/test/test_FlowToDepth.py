import tensorflow as tf
import numpy as np
import sys
print(sys.path)
sys.path.insert(0,'../python')
from tfspecialops import tfspecialops as ops
import h5py

#USE_GPUS = sorted(set((False, tf.test.is_gpu_available())))
TYPES = (np.float32, np.float64)

USE_GPUS = [False]
TYPES = (np.float32, )

class FlowToDepthTest(tf.test.TestCase):

    def _load_blendswapdata(self):
        image_pair = None
        depth = None
        flow = None
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

                tmp = g['flow'][:][np.newaxis,:,:,:]
                if flow is None:
                    flow = tmp.copy()
                else:
                    flow = np.concatenate((flow, tmp))

                tmp = g['intrinsics'][:][np.newaxis,:]
                if intrinsics is None:
                    intrinsics = tmp.copy()
                else:
                    intrinsics = np.concatenate((intrinsics, tmp))
                
                tmp = g['motion'][:][np.newaxis,:]
                if motion is None:
                    motion = tmp.copy()
                else:
                    motion = np.concatenate((motion, tmp))

        return {'image_pair': image_pair,
                'depth': depth,
                'flow': flow,
                'intrinsics': intrinsics,
                'motion': motion }


    def _test_blendswapdata(self,data):
        depth_tensor = ops.flow_to_depth(
                inverse_depth=True,
                normalized_flow=True,
                flow=data['flow'], 
                intrinsics=data['intrinsics'], 
                rotation=data['motion'][:,0:3], 
                translation=data['motion'][:,3:])
        depth = depth_tensor.eval()
        import ijremote
        ijremote.setImage("image_pair", data['image_pair'])
        ijremote.setImage("depth", depth)
        ijremote.setImage("depth_gt", data['depth'])
        self.assertAllClose(depth, data['depth'])

                    
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

    def test_shape(self):
        with self.test_session(use_gpu=False, force_gpu=False):
            depth1 = np.empty((8,40,31))
            depth2 = np.empty((8,1,40,31))
            depth3 = np.empty((2,2,2,40,31))
            depth_inputs = (depth1, depth2, depth3)
            intrinsics = np.empty((8,4))
            rotation = np.empty((8,3))
            translation = np.empty((8,3))

            expected_shape = [8,2,40,31]

            for depth in depth_inputs:
                output_tensor = ops.depth_to_flow(
                        depth=depth,
                        intrinsics=intrinsics,
                        rotation=rotation,
                        translation=translation )
                out_shape = output_tensor.get_shape().as_list()
                self.assertAllEqual(out_shape, expected_shape)


                



if __name__ == '__main__':
    tf.test.main()


