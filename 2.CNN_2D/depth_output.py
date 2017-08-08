
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.insert(0, './layers1.0/')
import skimage.io as io
import scipy.misc
from numpy import newaxis
import matplotlib.pyplot as plt
from PIL import Image

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


import skimage.io as io
import scipy.misc
import tensorflow as tf
import os
from numpy import newaxis
import matplotlib.pyplot as plt
import sys
import argparse
CHECKPOINT_DIR=param.TRAIN_LOG
#NUM_EXAMPLES_PER_EPOCH_FOR_EVAL=2
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL=param.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
BATCH_SIZE=param.BATCH_SIZE
MOVING_AVERAGE_DECAY=param.MOVING_AVERAGE_DECAY
TEST_LOG=param.TEST_LOG
EVAL_RUN_ONCE=True
#EVAL_RUN_ONCE=param.EVAL_RUN_ONCE
EVAL_INTERVAL_SECS=param.EVAL_INTERVAL_SECS
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
cur_dir="./pictures/"
import moviepy.editor as mpy
def npy_to_gif(npy, filename):
    clip = mpy.ImageSequenceClip(list(npy), fps=10)
    clip.write_gif(filename)

def eval_once(result,depths,config,saver):
  
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
            total_sample_count = num_iter * BATCH_SIZE
            step = 0
            print('%s: starting predictions on (%s).' % (datetime.now(), NUM_EXAMPLES_PER_EPOCH_FOR_EVAL))
            start_time = time.time()
            while step < num_iter and not coord.should_stop():   
                
                rs,dpth=sess.run([result,depths])
                mind=np.amin(dpth[0,:,:,0])
                maxd=np.amax(dpth[0,:,:,0])
                print("mind ", mind)
                print("maxd ", maxd)
                
                min1=np.amin(rs[0,:,:,0])
                max1=np.amax(rs[0,:,:,0])
                print("result min ", min1)
                print("result max ", max1) 
                print(rs.shape)
                name1=cur_dir+"res"+str(step)+".png"
                scipy.misc.toimage(rs[0,:,:,0], cmin=min1, cmax=max1).save(name1)
    
                name2=cur_dir+"gt"+str(step)+".png" 
                scipy.misc.toimage(dpth[0,:,:,0], cmin=mind, cmax=maxd).save(name2)
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
    

    result = model.inference(images,True)
    result = lss.inverse(result)
    depths = lss.inverse(depths)
    #viz1=tf.slice(result,(0,0,0,0), (1,192,256,1)) 
    #viz2=tf.slice(result,(0,0,0,1), (1,192,256,1))
    #gt1=tf.slice(depths,(0,0,0,0), (1,192,256,1))
    #gt2=tf.slice(depths,(0,0,0,1), (1,192,256,1))
    #zero=tf.zeros_like(gt1)
    #mask = tf.not_equal(gt1, zero)
    #mask=tf.cast(mask,tf.float32) 
    #viz11=tf.multiply(mask,viz)
   # mask2 = tf.not_equal(gt2, zero)
    #mask2=tf.cast(mask2,tf.float32) 
    #viz12=tf.multiply(mask2,viz2) 
    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        model.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    
    config = tf.ConfigProto(gpu_options=gpu_options)
    while True:
      print('Start depth output')   
      eval_once(result,depths,config,saver)
      if EVAL_RUN_ONCE:
        print('end of depth output')
        break
      time.sleep(EVAL_INTERVAL_SECS)


def main(argv=None): 
  evaluate()


if __name__ == '__main__':
  tf.app.run() 
