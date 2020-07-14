from __future__ import division
import random
import os
import cPickle
from keras_prnet import config

# pass the settings in the config object
C = config.Config()
os.environ["CUDA_VISIBLE_DEVICES"] = '0,2'
C.init_lr = 1e-4
C.alpha = 0.999  # update rate of weight moving average
C.add_epoch = 0

# define the path for loading the initialized weight files
weight_path = 'data/models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

# define output path for weight files
out_path = 'output/valmodels/prnet'
if not os.path.exists(out_path):
    os.makedirs(out_path)

# get the training data
cache_path = 'data/cache/cityperson/train'
with open(cache_path, 'rb') as fid:
    train_data = cPickle.load(fid)
num_imgs_train = len(train_data)
random.shuffle(train_data)
print 'num of training samples: {}'.format(num_imgs_train)

from keras_prnet.model.model_prnet import Model_prnet
model = Model_prnet()
model.initialize(C)
model.creat_model(C, train_data, phase = 'train',wei_mov_ave=True)
model.train_model(C, weight_path, out_path)
