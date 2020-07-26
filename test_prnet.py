from __future__ import division

import os
import cPickle

from keras_prnet import config
from keras_prnet.model.model_prnet import Model_prnet
from evaluation.eval_script.coco import COCO
from evaluation.eval_script.eval_MR_multisetup import COCOeval

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# Set model path in config object.
model_config = config.Config()
model_path = './data/models/PRNet_city.hdf5'
model_epoch = 82

# Define output path for detection results.
output_path = 'output/valresults/prnet'
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Get test data.
cache_path = 'data/cache/cityperson/val'
with open(cache_path, 'rb') as fid:
    val_data = cPickle.load(fid)
num_imgs = len(val_data)
print('num of val samples: {}'.format(num_imgs))

# Create and evaluate PRNet network.
model = Model_prnet()
model.initialize(model_config)
model.creat_model(model_config, val_data, phase='inference')
model.test_model(model_config, model_epoch, val_data, model_path, output_path)

# Get evaluation results.
ann_type = 'bbox'
ann_file = 'evaluation/val_gt.json'
dt_path = os.path.join(output_path)
res_file = os.path.join(dt_path, 'val_det_epoch%s.json' % model_epoch)
summary_file = open(os.path.join(dt_path, 'results_epoch%s.txt' % model_epoch), 'w')
for id_setup in range(0, 4):
    cocoGt = COCO(ann_file)
    cocoDt = cocoGt.loadRes(res_file)
    imgIds = sorted(cocoGt.getImgIds())
    cocoEval = COCOeval(cocoGt, cocoDt, ann_type)
    cocoEval.params.imgIds  = imgIds
    cocoEval.evaluate(id_setup)
    cocoEval.accumulate()
    cocoEval.summarize(id_setup, summary_file)
