from __future__ import division
import os
import cPickle
from keras_alfnet import config

# pass the settings in the config object
C = config.Config()
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

if C.network=='resnet50':
    w_path = 'data/models/city_res50_%sstep.hdf5' % (C.steps)
elif C.network=='mobilenet' and C.steps in [1,2]:
    w_path = 'data/models/city_mobnet_%sstep.hdf5'% (C.steps)
else:
    raise NotImplementedError('Not support network: {}'.format(C.network))

# define output path for detection results
out_path = 'output/valresults/%s/%dstep' % (C.network, C.steps)
if not os.path.exists(out_path):
    os.makedirs(out_path)

# get the test data
cache_path = 'data/cache/cityperson/val'
with open(cache_path, 'rb') as fid:
	val_data = cPickle.load(fid)
num_imgs = len(val_data)
print 'num of val samples: {}'.format(num_imgs)

C.random_crop = (1024, 2048)
# define the ALFNet network
if C.steps == 1:
    from keras_alfnet.model.model_1step import Model_1step
    model = Model_1step()
elif C.steps == 2:
    from keras_alfnet.model.model_2step import Model_2step
    model = Model_2step()
elif C.steps == 3:
    from keras_alfnet.model.model_3step import Model_3step
    model = Model_3step()
else:
    raise NotImplementedError('Not implement {} or more steps'.format(C.steps))
model.initialize(C)
model.creat_model(C, val_data, phase='inference')
model.test_model(C, val_data, w_path, out_path)