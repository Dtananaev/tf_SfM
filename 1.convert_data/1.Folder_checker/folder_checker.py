import numpy as np
import os

list_train='../../0.dataset_split/list_train.txt'
list_test='../../0.dataset_split/list_test.txt'
dataset='/misc/lmbraid12/datasets/public/sun3d/data' # specify where is your SUN3D dataset

def read_file(textfile):
  '''Read txt file and output array of strings line by line '''
  with open(textfile) as f:
    result = f.read().splitlines()
  return result

def write_txtfile(data,path): 
    
   '''Write array of strings to the file specified in path'''
   with open(path, 'w') as f:
    for item in data:
        f.write("%s\n" % item)
        
        
def make_list(list_train,dataset):
  data=read_file(list_train);
  result=[]
  for i in range(0,len(data)):
    path=dataset+"/"+data[i]
    im=os.path.isdir(path+"/image/")
    dp=os.path.isdir(path+"/depth/")
    extr=os.path.isdir(path+"/extrinsics/")
    intr=os.path.exists(path+"/intrinsics.txt")
    if(im and dp and extr and intr):
      result.append(data[i])
  return result

train=make_list(list_train,dataset)
test=make_list(list_test,dataset)
write_txtfile(train,"./train_list.txt")
write_txtfile(test,"./test_list.txt")