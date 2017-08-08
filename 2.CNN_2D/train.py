#
# File: train.py
# Date:21.01.2017
# Author: Denis Tananaev
# 
#

 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time

import tensorflow as tf
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

import model
import param
import data
TRAIN_LOG=param.TRAIN_LOG
BATCH_SIZE=param.BATCH_SIZE
NUM_ITER=param.NUM_ITER
NUM_PREPROCESS_THREADS=param.NUM_PREPROCESS_THREADS
LOG_DEVICE_PLACEMENT=param.LOG_DEVICE_PLACEMENT
def train():
  """Train SUN3D for a number of steps."""
  with tf.Graph().as_default(),tf.device('/gpu:0'):
    global_step = tf.contrib.framework.get_or_create_global_step()

    # Get images and labels for SUN3D.
    images, gtdepths, gttransforms = data.read_dataset(eval_data=False)

    pdepth, ptransforms = model.inference(images,True)
    
    # Calculate loss.
    loss = model.loss(pdepth,gtdepths,gttransforms,ptransforms)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = model.train(loss, global_step)
    
    config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=LOG_DEVICE_PLACEMENT,intra_op_parallelism_threads=NUM_PREPROCESS_THREADS)
    config.gpu_options.allow_growth = True
    #config.gpu_options.per_process_gpu_memory_fraction = 0.4

    class _LoggerHook(tf.train.SessionRunHook):
      """Logs loss and runtime."""

      def begin(self):
        self._step = -1

      def before_run(self, run_context):
        self._step += 1
        self._start_time = time.time()
        return tf.train.SessionRunArgs(loss)  # Asks for loss value.

      def after_run(self, run_context, run_values):
        duration = time.time() - self._start_time
        loss_value = run_values.results
        if self._step % 10 == 0:
          num_examples_per_step = BATCH_SIZE
          examples_per_sec = num_examples_per_step / duration
          sec_per_batch = float(duration)
     
          format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print (format_str % (datetime.now(), self._step, loss_value,
                               examples_per_sec, sec_per_batch))
          
    with tf.train.MonitoredTrainingSession( save_checkpoint_secs=3600,checkpoint_dir=TRAIN_LOG,
hooks=[tf.train.StopAtStepHook(last_step=NUM_ITER),tf.train.NanTensorHook(loss),_LoggerHook()],config=config) as mon_sess:
        while not mon_sess.should_stop():
            print(mon_sess.run(loss))
            mon_sess.run(train_op )

            
def main(argv=None):
  train()


if __name__ == '__main__':
  tf.app.run() 
 
