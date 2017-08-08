import tensorflow as tf
import numpy as np
import sys
sys.path.insert(0,'../python')
from tfspecialops import tfspecialops as ops

USE_GPUS = sorted(set((False, tf.test.is_gpu_available())))
TYPES = (np.float32, np.float64)


class MySqrTest(tf.test.TestCase):

    def _testOp(self):
        A = np.array([1,2,3,4,5], dtype=np.float32)
        result = ops.my_sqr(A)
        self.assertAllEqual(result.eval(), [1, 4, 9, 16, 25])

    def testOp(self):
        for use_gpu in USE_GPUS:
            with self.test_session(use_gpu=use_gpu, force_gpu=use_gpu):
                self._testOp()


    def _testGrad(self):
        for dtype in TYPES:
            A = np.array([1,2,3,4,5], dtype=dtype)
            shape = A.shape
            data = tf.constant(A)
            output = ops.my_sqr(data)
            err = tf.test.compute_gradient_error(
                    data, shape,
                    output, shape,
                    x_init_value = A )
            self.assertLess(err, 1e-3)

    def testGrad(self):
        for use_gpu in USE_GPUS:
            with self.test_session(use_gpu=use_gpu, force_gpu=use_gpu):
                self._testGrad()



if __name__ == '__main__':
    tf.test.main()
