import tensorflow as tf
import numpy as np
import sys
print(sys.path)
sys.path.insert(0,'../python')
from tfspecialops import tfspecialops as ops
np.set_printoptions(linewidth=160)

USE_GPUS = sorted(set((False, tf.test.is_gpu_available())))
TYPES = (np.float32, np.float64)

gpu_options = tf.GPUOptions()
gpu_options.per_process_gpu_memory_fraction=0.2
session_params = {'config': tf.ConfigProto(gpu_options=gpu_options) }

USE_GPUS = [False]
#USE_GPUS = [True]
# TYPES = (np.float64,)
# TYPES = (np.float32,)

class AngleAxisToRotationMatrixTest(tf.test.TestCase):

                    
        
    def _test_random_rotation(self,dtype):
        aa = np.random.rand(4,3).astype(dtype)

        aa_tensor = tf.constant(aa)
        R_tensor = ops.angle_axis_to_rotation_matrix(aa_tensor)
        aa2_tensor = ops.rotation_matrix_to_angle_axis(R_tensor)
        aa2 = aa2_tensor.eval()
        err = np.sum(np.abs(aa-aa2))
        print('error',err,flush=True)
        self.assertLess(err, 1e-5)

    def test_random_rotation(self):
        for use_gpu in USE_GPUS:
            for dtype in TYPES:
                print(use_gpu, dtype, flush=True)
                with self.test_session(use_gpu=use_gpu, force_gpu=use_gpu, **session_params):
                    self._test_random_rotation(dtype)



    def _test_zero_rotation(self,dtype):
        aa = np.zeros(3).astype(dtype)

        aa_tensor = tf.constant(aa)
        R_tensor = ops.angle_axis_to_rotation_matrix(aa_tensor)
        R = R_tensor.eval()
        I = np.eye(3)
        err = np.sum(np.abs(R-I))
        print('error',err,flush=True)
        self.assertLess(err, 1e-5)

    def test_zero_rotation(self):
        for use_gpu in USE_GPUS:
            for dtype in TYPES:
                print(use_gpu, dtype, flush=True)
                with self.test_session(use_gpu=use_gpu, force_gpu=use_gpu, **session_params):
                    self._test_zero_rotation(dtype)


    def _test_grad(self,dtype):
        aa = np.random.rand(5,3).astype(dtype)

        shape = aa.shape
        data = tf.constant(aa)
        output = ops.angle_axis_to_rotation_matrix(data)
        err = tf.test.compute_gradient_error(
                data, shape,
                output, output.get_shape().as_list(),
                x_init_value = aa )
        print('error',err,flush=True)
        self.assertLess(err, 1e-4)
        grad = tf.test.compute_gradient(
                data, shape,
                output, output.get_shape().as_list(),
                x_init_value = aa ,
                delta=0.0001
                )
        diff = np.abs(grad[0]-grad[1])
        # print('diff\n', diff)
        # for g in grad:
            # print(g[:,:])
            # print(g.shape)

    def test_grad(self):
        for use_gpu in USE_GPUS:
            for dtype in TYPES:
                print(use_gpu, dtype, flush=True)
                with self.test_session(use_gpu=use_gpu, force_gpu=use_gpu, **session_params):
                    self._test_grad(dtype)



    def _test_grad_zero_rotation(self,dtype):
        # aa = np.random.rand(1,3).astype(dtype)
        #aa = np.array([0,1,0],dtype=dtype)
        aa = np.array([0,0,0],dtype=dtype)

        shape = aa.shape
        data = tf.constant(aa)
        output = ops.angle_axis_to_rotation_matrix(data)
        err = tf.test.compute_gradient_error(
                data, shape,
                output, output.get_shape().as_list(),
                x_init_value = aa )
        print('error',err,flush=True)
        self.assertLess(err, 1e-5)
        grad = tf.test.compute_gradient(
                data, shape,
                output, output.get_shape().as_list(),
                x_init_value = aa ,
                delta=0.0001
                )
        diff = np.abs(grad[0]-grad[1])
        # print('diff\n', diff)
        # for g in grad:
            # print(g[:,:])
            # print(g.shape)

    def test_grad_zero_rotation(self):
        for use_gpu in USE_GPUS:
            for dtype in TYPES:
                print(use_gpu, dtype, flush=True)
                with self.test_session(use_gpu=use_gpu, force_gpu=use_gpu, **session_params):
                    self._test_grad_zero_rotation(dtype)


    def test_shape(self):
        with self.test_session(use_gpu=False, force_gpu=False, **session_params):
            input1 = np.empty((8,3))
            input2 = np.empty((2,4,3))
            input3 = np.empty((2,2,2,3))
            inputs = (input1, input2, input3)

            expected_shape = [8,3,3]

            for i in inputs:
                output_tensor = ops.angle_axis_to_rotation_matrix( i )
                out_shape = output_tensor.get_shape().as_list()
                self.assertAllEqual(out_shape, expected_shape)


                



if __name__ == '__main__':
    tf.test.main()



