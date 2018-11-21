from __future__ import division
import random
import os
import cPickle
from keras_alfnet import config

# pass the settings in the config object
C = config.Config()
os.environ["CUDA_VISIBLE_DEVICES"] = C.gpu_ids

C.init_lr = 1e-4
# define the path for loading the initialized weight files
if C.network=='resnet50':
    weight_path = 'data/models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
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

# define the ALFNet network and start training
if C.steps == 1:
    from keras_alfnet.model.model_1step import Model_1step
    model = Model_1step()
elif C.steps == 2:
    C.neg_overlap_step2 = 0.5
    C.pos_overlap_step2 = 0.7
    from keras_alfnet.model.model_2step import Model_2step
    model = Model_2step()
elif C.steps == 3:
    from keras_alfnet.model.model_3step import Model_3step
    model = Model_3step()
else:
    raise NotImplementedError('Not implement {} or more steps'.format(C.steps))
model.initialize(C)
model.creat_model(C, train_data, phase = 'train')
model.train_model(C, weight_path, out_path)