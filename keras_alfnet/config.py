
class Config:

	def __init__(self):
		self.gpu_ids = '0,1'
		self.onegpu = 10
		self.num_epochs = 150
		self.add_epoch = 0
		self.iter_per_epoch = 2000
		self.init_lr = 1e-4

		# setting for network architechture
		self.network = 'resnet50' # or 'mobilenet'
		self.steps = 2  # optionally, ALF steps can be 1,2,3,...

		# setting for data augmentation
		self.use_horizontal_flips = True
		self.brightness = (0.5, 2, 0.5)
		self.in_thre = 0.5
		self.scale = (0.3, 1.0)
		self.random_crop = (640, 1280)

		# image channel-wise mean to subtract, the order is BGR
		self.img_channel_mean = [103.939, 116.779, 123.68]

		# setting for scales
		self.anchor_box_scales = [[16, 24], [32, 48], [64, 80], [128, 160]]
		self.anchor_ratios = [[0.41], [0.41], [0.41], [0.41]]

		# scaling the stdev
		self.classifier_regr_std = [0.1, 0.1, 0.2, 0.2]

		# overlaps for ignore areas
		self.ig_overlap = 0.5
		# overlaps for different ALF steps
		self.neg_overlap_step1 = 0.3
		self.pos_overlap_step1 = 0.5
		self.neg_overlap_step2 = 0.4
		self.pos_overlap_step2 = 0.65
		self.neg_overlap_step3 = 0.5
		self.pos_overlap_step3 = 0.75

		# setting for inference
		self.scorethre= 0.1
		self.overlap_thresh = 0.5
		self.pre_nms_topN = 6000
		self.post_nms_topN = 100
		self.roi_stride= 16

