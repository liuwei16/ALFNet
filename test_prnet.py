from __future__ import division
import os
import cPickle
from keras_prnet import config
from evaluation.eval_script.coco import COCO
from evaluation.eval_script.eval_MR_multisetup import COCOeval

# pass the settings in the config object
C = config.Config()
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
epoch=82
w_path="./data/models/PRNet_city.hdf5"

# define output path for detection results
out_path = 'output/valresults/prnet'
if not os.path.exists(out_path):
    os.makedirs(out_path)

# get the test data
cache_path = 'data/cache/cityperson/val'
with open(cache_path, 'rb') as fid:
	val_data = cPickle.load(fid)
num_imgs = len(val_data)
print 'num of val samples: {}'.format(num_imgs)

# define the PRNet network
from keras_prnet.model.model_prnet import Model_prnet
model = Model_prnet()
model.initialize(C)
model.creat_model(C, val_data, phase='inference')
model.test_model(C, epoch, val_data, w_path, out_path)

# get valuation results
annType = 'bbox'
annFile = 'evaluation/val_gt.json'
main_path = out_path
dt_path = os.path.join(main_path)
resFile = os.path.join(dt_path,'val_det_epoch%s.json')%(epoch)
respath = os.path.join(dt_path,'results_epoch%s.txt')%(epoch)
res_file = open(respath, "w")
for id_setup in range(0,4):
   cocoGt = COCO(annFile)
   cocoDt = cocoGt.loadRes(resFile)
   imgIds = sorted(cocoGt.getImgIds())
   cocoEval = COCOeval(cocoGt,cocoDt,annType)
   cocoEval.params.imgIds  = imgIds
   cocoEval.evaluate(id_setup)
   cocoEval.accumulate()
   cocoEval.summarize(id_setup,res_file)
