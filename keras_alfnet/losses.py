from keras import backend as K

if K.image_dim_ordering() == 'tf':
	import tensorflow as tf

epsilon = 1e-4
def regr_loss(y_true, y_pred):
	positives = y_true[:, :, 0]
	n_positive = tf.reduce_sum(positives)
	absolute_loss = tf.abs(y_true[:,:,1:] - y_pred)
	square_loss = 0.5 * (y_true[:,:,1:] - y_pred) ** 2
	l1_loss = tf.where(tf.less(absolute_loss, 1.0), square_loss, absolute_loss - 0.5)
	localization_loss = tf.to_float(tf.reduce_sum(l1_loss, axis=-1))
	loc_loss = tf.reduce_sum(localization_loss * positives, axis=-1)/ tf.maximum(1.0, n_positive)
	return loc_loss

def occlusion_regr_loss(y_true, y_pred):
	weights = y_true[:, :, 0]
	positives=y_true[:, :, 1]
	n_positive = tf.reduce_sum(positives)
	absolute_loss = tf.abs(y_true[:,:,2:] - y_pred)
	square_loss = 0.5 * (y_true[:,:,2:] - y_pred) ** 2
	l1_loss = tf.where(tf.less(absolute_loss, 1.0), square_loss, absolute_loss - 0.5)
	localization_loss = tf.to_float(tf.reduce_sum(l1_loss, axis=-1))
	loc_loss = tf.constant(4.0)*tf.reduce_sum(localization_loss * weights, axis=-1)/ tf.maximum(1.0, n_positive)
	return loc_loss

def cls_loss(y_true, y_pred):
	positives = y_true[:, :, 0]
	negatives = y_true[:, :, 1]
	valid = positives + negatives
	classification_loss = valid * K.binary_crossentropy(y_pred[:, :, 0], positives)
	# firstly compute the focal weight
	foreground_alpha = positives * tf.constant(0.25)
	background_alpha = negatives * tf.constant(0.75)
	foreground_weight = foreground_alpha * (tf.constant(1.0) - y_pred[:, :, 0]) ** tf.constant(2.0)
	background_weight = background_alpha * y_pred[:, :, 0] ** tf.constant(2.0)
	focal_weight = foreground_weight + background_weight
	assigned_boxes = tf.reduce_sum(positives)
	class_loss = tf.reduce_sum(classification_loss * focal_weight, axis=-1) / tf.maximum(1.0, assigned_boxes)

	return class_loss
