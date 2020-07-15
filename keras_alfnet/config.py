
class Config:

	def __init__(self):
		self.gpu_ids = '0,2'
		self.onegpu = 4
		self.num_epochs = 100
		self.add_epoch = 0
		self.iter_per_epoch = 2000
		self.init_lr = 1e-4

		# setting for data augmentation
		self.use_horizontal_flips = True
		self.brightness = (0.5, 2, 0.5)
		self.in_thre = 0.5
		self.scale = (0.3, 1.0)
		self.random_crop = (1024, 2048)
		self.size_train = (1024, 2048)

		# image channel-wise mean to subtract, the order is BGR
		self.img_channel_mean = [103.939, 116.779, 123.68]

		# setting for scales
		self.anchor_box_scales = [[16, 24], [32, 48], [64, 80], [128, 160]]
		self.anchor_ratios = [[0.41], [0.41], [0.41], [0.41]]

		# scaling the stdev
		self.classifier_regr_std = [0.1, 0.1, 0.2, 0.2]

		# overlaps for ignore areas
		self.ig_overlap = 0.5
		# labeling stratety threshold
		self.neg_overlap_ve = 0.3
		self.pos_overlap_ve = 0.5
		self.neg_overlap_fr = 0.5
		self.pos_overlap_fr = 0.7

		# setting for inference
		self.scorethre= 0.05
		self.overlap_thresh = 0.5 #threshold in nms stage
		self.pre_nms_topN = 6000
		self.post_nms_topN = 100
		self.roi_stride= 16

