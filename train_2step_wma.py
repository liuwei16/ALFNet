# --------------------------------------------------------
# train_2step_wma
# Written by Liu Wei
#
# Example of training ALFNet-2s with the strategy of weight moving average (WMA) proposed in Mean-Teacher
# ref: https://arxiv.org/abs/1703.01780
#
# WMA helps us achieve more stable results
# --------------------------------------------------------
from __future__ import division
import random
import os
import cPickle
from keras_alfnet import config

# pass the settings in the config object
C = config.Config()
os.environ["CUDA_VISIBLE_DEVICES"] = C.gpu_ids

C.init_lr = 1e-4
C.alpha = 0.999  # update rate of weight moving average

# define the path for loading the initialized weight files
if C.network=='resnet50':
    weight_path = 'data/models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
    # weight_path = 'output/valmodels/resnet50/2step/0.0001/resnet_e54_l0.0845779655501.hdf5'
elif C.network=='mobilenet':
    weight_path = 'data/models/mobilenet_1_0_224_tf_no_top.h5'
else:
    raise NotImplementedError('Not support network: {}'.format(C.network))

# define output path for weight files
out_path = 'output/valmodels/%s/%dstep/%s' % (C.network, C.steps, C.init_lr)
if not os.path.exists(out_path):
    os.makedirs(out_path)

# get the training data
cache_path = 'data/cache/cityperson/train'
with open(cache_path, 'rb') as fid:
    train_data = cPickle.load(fid)
num_imgs_train = len(train_data)
random.shuffle(train_data)
print 'num of training samples: {}'.format(num_imgs_train)

# define ALFNet-2s and start training
C.neg_overlap_step2 = 0.5
C.pos_overlap_step2 = 0.7
from keras_alfnet.model.model_2step import Model_2step
model = Model_2step()
model.initialize(C)
model.creat_model(C, train_data, phase = 'train',wei_mov_ave=True)
model.train_model_wma(C, weight_path, out_path)
