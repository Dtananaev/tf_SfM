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
#TYPES = (np.float32,)

class RotationMatrixToAngleAxisTest(tf.test.TestCase):

                    
        
    def _test_zero_rotation(self,dtype):
        R = np.eye(3).astype(dtype)

        R_tensor = tf.constant(R)
        aa_tensor = ops.rotation_matrix_to_angle_axis(R_tensor)
        aa = aa_tensor.eval()
        err = np.sum(np.abs(aa))
        print('error',err,flush=True)
        self.assertLess(err, 1e-5)

    def test_zero_rotation(self):
        for use_gpu in USE_GPUS:
            for dtype in TYPES:
                print(use_gpu, dtype, flush=True)
                with self.test_session(use_gpu=use_gpu, force_gpu=use_gpu, **session_params):
                    self._test_zero_rotation(dtype)



    def _test_grad_random_rotation(self,dtype):
        R = ops.angle_axis_to_rotation_matrix(np.random.rand(3).astype(dtype)).eval()
        shape = R.shape
        data = tf.constant(R)
        output = ops.rotation_matrix_to_angle_axis(data)
        err = tf.test.compute_gradient_error(
                data, shape,
                output, output.get_shape().as_list(),
                x_init_value = R )
        print('error',err,flush=True)
        self.assertLess(err, 1e-4)
        grad = tf.test.compute_gradient(
                data, shape,
                output, output.get_shape().as_list(),
                x_init_value = R ,
                delta=0.0001
                )
        diff = np.abs(grad[0]-grad[1])
        # print('diff\n', diff, "\n\n\n")
        # for g in grad:
            # print(g[:,:])
            # print(g.shape)

    def test_grad_random_rotation(self):
        for use_gpu in USE_GPUS:
            for dtype in TYPES:
                print(use_gpu, dtype, flush=True)
                with self.test_session(use_gpu=use_gpu, force_gpu=use_gpu, **session_params):
                    self._test_grad_random_rotation(dtype)


    def _test_grad_zero_rotation(self,dtype):

        # numerical differentiation does not work here with the identity because
        # the forward op limits the trace to be at most 3.
        # -> make the diagonal elements smaller
        R = np.eye(3).astype(dtype)
        delta = 0.0001
        R[0,0] -= delta
        R[1,1] -= delta
        R[2,2] -= delta

        shape = R.shape
        data = tf.constant(R)
        output = ops.rotation_matrix_to_angle_axis(data)
        numerical_grad = tf.test.compute_gradient(
                data, shape,
                output, output.get_shape().as_list(),
                x_init_value = R ,
                delta=delta
                )[1]

        # compute the theoretical gradient with a proper identity matrix
        R = np.eye(3).astype(dtype)
        data2 = tf.constant(R)
        output2 = ops.rotation_matrix_to_angle_axis(data2)
        theoretical_grad = tf.test.compute_gradient(
                data2, shape,
                output2, output2.get_shape().as_list(),
                x_init_value = R ,
                )[0]

        print('theoretical_grad\n', theoretical_grad)
        print('numerical_grad\n', numerical_grad)
        diff = np.abs(numerical_grad-theoretical_grad)
        err = np.sum(diff)
        self.assertLess(err, 1e-3)

    def test_grad_zero_rotation(self):
        for use_gpu in USE_GPUS:
            for dtype in TYPES:
                print(use_gpu, dtype, flush=True)
                with self.test_session(use_gpu=use_gpu, force_gpu=use_gpu, **session_params):
                    self._test_grad_zero_rotation(dtype)


    def test_shape(self):
        with self.test_session(use_gpu=False, force_gpu=False, **session_params):
            input1 = np.empty((8,3,3))
            input2 = np.empty((2,4,3,3))
            input3 = np.empty((2,2,2,3,3))
            inputs = (input1, input2, input3)

            expected_shape = [8,3]

            for i in inputs:
                output_tensor = ops.rotation_matrix_to_angle_axis( i )
                out_shape = output_tensor.get_shape().as_list()
                self.assertAllEqual(out_shape, expected_shape)


                



if __name__ == '__main__':
    tf.test.main()



