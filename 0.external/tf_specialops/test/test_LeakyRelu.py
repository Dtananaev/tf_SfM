import tensorflow as tf
import numpy as np
import sys
print(sys.path)
sys.path.insert(0,'../python')
from tfspecialops import tfspecialops as ops
np.set_printoptions(linewidth=160)

USE_GPUS = sorted(set((False, tf.test.is_gpu_available())))
TYPES = (np.float32, np.float64)

#USE_GPUS = [False]
#USE_GPUS = [True]
#TYPES = (np.float64,)
#TYPES = (np.float32,)

class LeakyReluTest(tf.test.TestCase):

                    
        
    def _test_grad(self,dtype):
        A = np.random.rand(9).astype(dtype) -0.5
        shape = A.shape
        data = tf.constant(A)
        output = ops.leaky_relu(input=data, leak=0.2)
        #print(A)
        #print(output.eval())
        err = tf.test.compute_gradient_error(
                data, shape,
                output, output.get_shape().as_list(),
                x_init_value = A )
        print('error',err,flush=True)
        self.assertLess(err, 1e-3)
        grad = tf.test.compute_gradient(
                data, shape,
                output, output.get_shape().as_list(),
                x_init_value = A, 
                delta=0.1)
        for g in grad:
            print(g)
            print(g.shape)

    def test_grad(self):
        for use_gpu in USE_GPUS:
            for dtype in TYPES:
                print(use_gpu, dtype, flush=True)
                with self.test_session(use_gpu=use_gpu, force_gpu=use_gpu):
                    self._test_grad(dtype)


    def test_shape(self):
        with self.test_session(use_gpu=False, force_gpu=False):
            input1 = np.empty((8,40,31))
            input2 = np.empty((8,1,40,31))
            input3 = np.empty((2,2,2,40,31))
            inputs = (input1, input2, input3)

            for i in inputs:
                output_tensor = ops.replace_nonfinite( input=i )
                out_shape = output_tensor.get_shape().as_list()
                self.assertAllEqual(out_shape, i.shape)


                



if __name__ == '__main__':
    tf.test.main()




