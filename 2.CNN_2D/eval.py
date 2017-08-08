
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.insert(0, '../0.layers/')
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
from datetime import datetime
import math
import os.path
import time
import numpy as np
import tensorflow as tf
import param
import model
import data
import evalfunct 
import losses as lss
IMAGE_SIZE_W=param.IMAGE_SIZE_W
IMAGE_SIZE_H=param.IMAGE_SIZE_H
CHECKPOINT_DIR=param.TRAIN_LOG
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL=param.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
BATCH_SIZE=param.BATCH_SIZE
MOVING_AVERAGE_DECAY=param.MOVING_AVERAGE_DECAY
TEST_LOG=param.TEST_LOG
EVAL_RUN_ONCE=param.EVAL_RUN_ONCE
EVAL_INTERVAL_SECS=param.EVAL_INTERVAL_SECS


def eval_once(result,config,saver, summary_writer, scale_inv_error, L1_relative_error,L1_inverse_error,L1_transform, summary_op):
  
    with tf.Session(config=config) as sess:
        ckpt = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
             #Assuming model_checkpoint_path looks something like:
            # extract global_step from it.
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            print('Checkpoint is loaded') 
        else:
            print('No checkpoint file found')
            return
        # Start the queue runners.
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

            num_iter = int(math.ceil(NUM_EXAMPLES_PER_EPOCH_FOR_EVAL / BATCH_SIZE))
            # Counts the number of correct predictions.
            count_scale_inv_error = 0.0
            count_L1_relative_error = 0.0
            count_L1_inverse_error = 0.0
            count_L1_transform_error = 0.0
            
            total_sample_count = num_iter * BATCH_SIZE
            step = 0
            error=0
            print('%s: starting evaluation on (%s).' % (datetime.now(), NUM_EXAMPLES_PER_EPOCH_FOR_EVAL))
            start_time = time.time()
            while step < num_iter and not coord.should_stop():   
                
                scale_inv_error_res,L1_relative_error_res,L1_inverse_error_res ,L1_transform_res=sess.run([scale_inv_error,L1_relative_error,L1_inverse_error,L1_transform])
               
                count_scale_inv_error += scale_inv_error_res
                count_L1_relative_error += L1_relative_error_res
                count_L1_inverse_error += L1_inverse_error_res       
                count_L1_transform_error+=L1_transform_res
                step += 1
                print(step)
                if step % 20 == 0:
                    duration = time.time() - start_time
                    sec_per_batch = duration / 20.0
                    examples_per_sec = BATCH_SIZE / sec_per_batch
                    print('%s: [%d batches out of %d] (%.1f examples/sec; %.3f'
                'sec/batch)' % (datetime.now(), step, num_iter,
                                examples_per_sec, sec_per_batch))
                    start_time = time.time()


            result_scale_inv_error = count_scale_inv_error / num_iter
            result_L1_relative_error = count_L1_relative_error / num_iter
            result_L1_inverse_error = count_L1_inverse_error / num_iter      
            result_L1_transform_error = count_L1_transform_error / num_iter
            print('%s: sc-inv = %.4f L1-rel = %.4f L1-inv = %.4f L1-transform %.4f  [%d examples]' %
            (datetime.now(), result_scale_inv_error, result_L1_relative_error,result_L1_inverse_error,result_L1_transform_error, total_sample_count))

            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='sc-inv ', simple_value=result_scale_inv_error)
            summary.value.add(tag=' L1-rel ', simple_value=result_L1_relative_error)
            summary.value.add(tag=' L1-inv ', simple_value=result_L1_inverse_error) 
            summary.value.add(tag=' L1-transform ', simple_value=result_L1_transform_error)              
            summary_writer.add_summary(summary, global_step)

        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)
    
def evaluate():
  """Eval SUN3D for a number of steps."""
  with tf.Graph().as_default() as g:
    # Get images and labels for SUN3D.
    images, depths,transforms = data.read_dataset(eval_data=True)
    # Build a Graph that computes the logits predictions from the
    # inference model.
    
    result, resulttransform  = model.inference(images,False)
    depths = lss.inverse(depths)
    result = lss.inverse(result)
    depths=tf.slice(depths,(0, 0, 0, 0), (BATCH_SIZE,IMAGE_SIZE_H, IMAGE_SIZE_W,1))
    
    # Calculate predictions.
    scale_inv_error=evalfunct.scinv(result,depths)
    L1_relative_error=evalfunct.L1rel(result,depths)
    L1_inverse_error=evalfunct.L1inv(result,depths)
    L1_transform= tf.reduce_mean(tf.abs(resulttransform-transforms))
    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        model.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(TEST_LOG, g)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    config = tf.ConfigProto(gpu_options=gpu_options)
    while True:
      print('Start evaluation')   
      eval_once(result,config,saver, summary_writer,  scale_inv_error, L1_relative_error,L1_inverse_error,L1_transform, summary_op)
      if EVAL_RUN_ONCE:
        print('end of evaluation')
        break
      time.sleep(EVAL_INTERVAL_SECS)


def main(argv=None): 
  evaluate()


if __name__ == '__main__':
  tf.app.run() 
