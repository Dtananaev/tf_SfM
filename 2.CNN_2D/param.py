#
# File: param.py
# Date:21.01.2017
# Author: Denis Tananaev
# 
#
#the dataset file for training set

#SSD
tfrecords_train_folder ='/misc/scratchSSD/ummenhof/denis/dataset/train/'
tfrecords_test_folder ='/misc/scratchSSD/ummenhof/denis/dataset/test/'
#tfrecords_train_folder ='../1.convert_data/train/' # specify train set folder with  sequences/tf_records
#tfrecords_test_folder ='../1.convert_data/train/' # specify test set folder with sequences/tf_records
#the parameters of dataset
IMAGE_SIZE_W=256
IMAGE_SIZE_H=192
FLOAT16=False

#the parameters data upload
BATCH_SIZE=16
SEQUENCE_LEN=1
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN=91865
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 353
# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.0     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 9999999999.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.0001       # Initial learning rate.
WEIGHT_DECAY=0.00001
#training and test
NUM_ITER=9999999999
TRAIN_LOG="./log"
TEST_LOG="./eval"

#loading param for data
NUM_PREPROCESS_THREADS=8
NUM_READERS=2
INPUT_QUEUE_MEMORY_FACTOR=4
#additional
LOG_DEVICE_PLACEMENT=False
PRETRAINED_MODEL_CHECKPOINT_PATH=''
EVAL_RUN_ONCE=True
EVAL_INTERVAL_SECS=5*60 #5 minutes


