from .base_model import Base_model
from keras.optimizers import Adam
from keras.models import Model
from keras_prnet.parallel_model import ParallelModel
from keras.utils import generic_utils
from keras_prnet import losses as losses
from keras_prnet import bbox_process
from evaluation.eval_script.coco import COCO
from evaluation.eval_script.eval_MR_multisetup import COCOeval
from keras import backend as K
from keras.layers import *
from matplotlib import pyplot
from . import prnetmodel_create
import numpy as np
import time, os, cv2
#import matplotlib.pyplot as plt
import json
import os

class Model_prnet(Base_model):
	def name(self):
		return 'Model_prnet'
	def initialize(self, opt):
		Base_model.initialize(self,opt)
		# specify the training details
		self.cls_loss_r1 = []
		self.regr_loss_r1 = []
		self.cls_loss_r2 = []
		self.regr_loss_r2 = []
		self.losses = np.zeros((self.epoch_length, 4))
		self.optimizer = Adam(lr=opt.init_lr)
		print 'Initializing the {}'.format(self.name())

	def creat_model(self,opt,train_data, phase='train', wei_mov_ave = False):
		Base_model.create_base_model(self, opt,train_data, phase=phase, wei_mov_ave = wei_mov_ave)
		ve,fr = prnetmodel_create.create_PRNet(self.base_layers, self.num_anchors, trainable=True, steps=2)
		if wei_mov_ave:
			ve_tea, fr_tea = prnetmodel_create.create_PRNet(self.base_layers_tea, self.num_anchors, trainable=True, steps=2)
			self.model_tea = Model(self.img_input, ve_tea + fr_tea)
		if phase=='train':
			self.model_ve = Model(self.img_input, ve)
			self.model_fr = Model(self.img_input, fr)
			if self.num_gpus > 1:
				self.model_ve = ParallelModel(self.model_ve, int(self.num_gpus))
				self.model_fr = ParallelModel(self.model_fr, int(self.num_gpus))
			self.model_ve.compile(optimizer=self.optimizer, loss=[losses.cls_loss, losses.regr_loss],sample_weight_mode=None)
			self.model_fr.compile(optimizer=self.optimizer, loss=[losses.cls_loss, losses.occlusion_regr_loss],sample_weight_mode=None)
		self.model_all = Model(self.img_input, ve+fr)
	
	def train_model(self,opt, weight_path, out_path):
		self.model_all.load_weights(weight_path, by_name=True)
		self.model_tea.load_weights(weight_path, by_name=True)
		print 'load weights from {}'.format(weight_path)
		iter_num = 0
		start_time = time.time()
		for epoch_num in range(self.num_epochs):
			progbar = generic_utils.Progbar(self.epoch_length)
			print('Epoch {}/{}'.format(epoch_num + 1 + self.add_epoch, self.num_epochs + self.add_epoch))
			while True:
				try:
					X, Y, img_data = next(self.data_gen_train)
					loss_s1 = self.model_ve.train_on_batch(X, Y)
					self.losses[iter_num, 0] = loss_s1[1]
					self.losses[iter_num, 1] = loss_s1[2]
					ve_pred = self.model_ve.predict_on_batch(X)
					Y2 = bbox_process.get_target_fr(self.anchors, ve_pred[1], img_data, opt,
													 igthre=opt.ig_overlap, posthre=opt.pos_overlap_fr,
													 negthre=opt.neg_overlap_fr)
					loss_s2 = self.model_fr.train_on_batch(X, Y2)
					self.losses[iter_num, 2] = loss_s2[1]
					self.losses[iter_num, 3] = loss_s2[2]
					# apply weight moving average
					for l in self.model_tea.layers:
						weights_tea = l.get_weights()
						if len(weights_tea) > 0:
							weights_stu = self.model_all.get_layer(name=l.name).get_weights()
							weights_tea = [opt.alpha * w_tea + (1 - opt.alpha) * w_stu for (w_tea, w_stu) in
										   zip(weights_tea, weights_stu)]
							l.set_weights(weights_tea)

					iter_num += 1
					if iter_num % 20 == 0:
						progbar.update(iter_num,
									   [('cls1', np.mean(self.losses[:iter_num, 0])),
										('regr1', np.mean(self.losses[:iter_num, 1])),
										('cls2', np.mean(self.losses[:iter_num, 2])),
										('regr2', np.mean(self.losses[:iter_num, 3]))])
					if iter_num == self.epoch_length:
						cls_loss1 = np.mean(self.losses[:, 0])
						regr_loss1 = np.mean(self.losses[:, 1])
						cls_loss2 = np.mean(self.losses[:, 2])
						regr_loss2 = np.mean(self.losses[:, 3])
						total_loss = cls_loss1 + regr_loss1 + cls_loss2 + regr_loss2

						self.total_loss_r.append(total_loss)
						self.cls_loss_r1.append(cls_loss1)
						self.regr_loss_r1.append(regr_loss1)
						self.cls_loss_r2.append(cls_loss2)
						self.regr_loss_r2.append(regr_loss2)

						print('Total loss: {}'.format(total_loss))
						print('Elapsed time: {}'.format(time.time() - start_time))

						iter_num = 0
						start_time = time.time()

						if total_loss < self.best_loss:
							print('Total loss decreased from {} to {}, saving weights'.format(self.best_loss, total_loss))
							self.best_loss = total_loss
						if (epoch_num+self.add_epoch)>=60:
							self.model_tea.save_weights(
									os.path.join(out_path, 'resnet_e{}.hdf5'.format(epoch_num + 1 + self.add_epoch)))
						break
				except Exception as e:
					print ('Found,Exception: {}'.format(e))
					continue
			records = np.concatenate((np.asarray(self.total_loss_r).reshape((-1, 1)),
									  np.asarray(self.cls_loss_r1).reshape((-1, 1)),
									  np.asarray(self.regr_loss_r1).reshape((-1, 1)),
									  np.asarray(self.cls_loss_r2).reshape((-1, 1)),
									  np.asarray(self.regr_loss_r2).reshape((-1, 1))),
									 axis=-1)
			np.savetxt(os.path.join(out_path, 'records.txt'), np.array(records), fmt='%.6f')
		print('Training complete, exiting.') 
		
	def test_model(self,opt, epoch, val_data, weight_path, out_path):
		self.model_all.load_weights(weight_path, by_name=True)
		print 'load weights from {}'.format(weight_path)
		result_list = []
		start_time = time.time()
		time_all=0
		res_file = os.path.join(out_path, 'val_det_epoch%s.json')%(epoch)
		for f in range(len(val_data)):
			filepath = val_data[f]['filepath']
			print('{}/50'.format(f))
			frame_number = f + 1
			img = cv2.imread(filepath)
			#start_time = time.time()
			x_in = bbox_process.format_img(img, opt)
			Y = self.model_all.predict(x_in)
			#time_all+=(time.time()-start_time)
			proposals= bbox_process.pred_ve(self.anchors, Y[0], Y[1], opt)
			bbx, scores= bbox_process.pred_fr(proposals, Y[2], Y[3], opt) 
			f_res = np.repeat(frame_number, len(bbx), axis=0).reshape((-1, 1))
			bbx[:, [2, 3]] -= bbx[:, [0, 1]]
      
			for ind in range(len(bbx)):
					dic={}
					dic["image_id"]=frame_number
					dic["category_id"]=1
					dic["bbox"]=bbx[ind, :].tolist()
					score = scores[ind].tolist()
					dic["score"]= score[0]
					result_list.append(dic)
		with open(res_file, "w") as resfile:
				resfile.writelines(json.dumps(result_list))
   



