from __future__ import division
import os
from keras_alfnet import config
from keras_alfnet.model.model_2step import Model_2step

# pass the settings in the config object
C = config.Config()
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

# define paths for weight files and detection results
w_path = 'data/models/city_res50_2step.hdf5'
data_path = 'data/examples/'
val_data = os.listdir(data_path)
out_path = os.path.join(data_path,'detections')
if not os.path.exists(out_path):
    os.makedirs(out_path)

C.random_crop = (1024, 2048)
C.network = 'resnet50'
# define the ALFNet network
model = Model_2step()
model.initialize(C)
model.creat_model(C, val_data, phase='inference')
model.demo(C, val_data, w_path, out_path)