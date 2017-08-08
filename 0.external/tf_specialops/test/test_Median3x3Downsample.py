import tensorflow as tf
import numpy as np
import sys
sys.path.insert(0,'../python')
from tfspecialops import tfspecialops as ops

USE_GPUS = sorted(set((False, tf.test.is_gpu_available())))
TYPES = (np.float32, np.float64)


class Median3x3DownsampleTest(tf.test.TestCase):

    def _test_single_element(self, dtype):
        A = np.array([[1]], dtype=dtype)
        tensor = ops.median3x3_downsample(A)
        B = tensor.eval()
        #print(A,B)
        self.assertAllEqual(B, A)

    def test_single_element(self):
        for use_gpu in USE_GPUS:
            for dtype in TYPES:
                with self.test_session(use_gpu=use_gpu, force_gpu=use_gpu):
                    self._test_single_element(dtype=dtype)

    def test_gpu_equals_cpu(self):
        if not tf.test.is_gpu_available():
            return
        for dtype in TYPES:
            A = np.random.rand(10,13).astype(dtype)
            with self.test_session(use_gpu=False, force_gpu=False):
                tensor_cpu = ops.median3x3_downsample(A)
                result_cpu = tensor_cpu.eval()

            with self.test_session(use_gpu=True, force_gpu=True):
                tensor_gpu = ops.median3x3_downsample(A)
                result_gpu = tensor_cpu.eval()

            self.assertAllEqual(result_cpu, result_gpu)

    def _test_result_1d(self, dtype):
        A = np.array([[1, 2, 3, 4, 5]],dtype=dtype)
        tensor = ops.median3x3_downsample(A)
        B = tensor.eval()
        correct = np.array([[1, 3, 5]],dtype=dtype)
        self.assertAllEqual(B, correct)

        tensor = ops.median3x3_downsample(A.transpose())
        B = tensor.eval()
        self.assertAllEqual(B, correct.transpose())

    def test_result_1d(self):
        for use_gpu in USE_GPUS:
            for dtype in TYPES:
                with self.test_session(use_gpu=use_gpu, force_gpu=use_gpu):
                    self._test_result_1d(dtype=dtype)




if __name__ == '__main__':
    tf.test.main()

