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


class Warp2dTest(tf.test.TestCase):

    def _load_blendswapdata(self):
        image_pair = None
        flow = None

        with h5py.File('test_data/blendswap_test_scenes.h5','r') as f:
            for gname, g in f.items():
                tmp = g['image_pair'][:][np.newaxis,:,:,:]
                if image_pair is None:
                    image_pair = tmp.copy()
                else:
                    image_pair = np.concatenate((image_pair, tmp))

                tmp = g['flow'][:][np.newaxis,:,:,:]
                if flow is None:
                    flow = tmp.copy()
                else:
                    flow = np.concatenate((flow, tmp))

        return {'image_pair': image_pair,
                'flow': flow }


    def _test_blendswapdata(self,data):
        normal_tensor = ops.warp2d(
                normalized=True,
                border_mode='value',
                border_value=-1,
                input=data['image_pair'][:,3:,:,:], 
                displacements=data['flow'] )
        warped = normal_tensor.eval()
        if( warped.dtype == np.float32 ):
            import ijremote
            ijremote.setImage("image_pair", data['image_pair'][:,0:3,:,:])
            ijremote.setImage("warped_image", warped)

                    
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


