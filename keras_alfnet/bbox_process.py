from __future__ import division
import numpy as np
# import matplotlib.pyplot as plt
from .utils.cython_bbox import bbox_overlaps
from .utils.bbox import box_op
from bbox_transform import bbox_transform_inv, bbox_transform,clip_boxes
from nms_wrapper import nms

def format_img(img, C):
	""" formats the image channels based on config """
	img = img.astype(np.float32)
	img[:, :, 0] -= C.img_channel_mean[0]
	img[:, :, 1] -= C.img_channel_mean[1]
	img[:, :, 2] -= C.img_channel_mean[2]

	img = np.expand_dims(img, axis=0)
	return img

def filter_boxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep
def filter_negboxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((ws <= min_size) | (hs <= min_size))[0]
    return keep
def compute_targets(ex_rois, gt_rois, classifier_regr_std,std):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4

    targets = bbox_transform(ex_rois, gt_rois)
	# Optionally normalize targets by a precomputed mean and stdev
    if std:
		targets = targets/np.array(classifier_regr_std)
    return targets

# for training
def get_target_1st(all_anchors, regr_layer, img_data, C,roi_stride=10,igthre=0.5,posthre=0.7,negthre=0.5):
	A = np.copy(all_anchors[:,:4])
	y_cls_batch, y_regr_batch = [], []
	for i in range(regr_layer.shape[0]):
		gta = np.copy(img_data[i]['bboxes'])
		num_bboxes = len(gta)
		ignoreareas = img_data[i]['ignoreareas']
		proposals = np.ones_like(all_anchors)
		bbox_deltas = regr_layer[i,:,:]
		bbox_deltas = bbox_deltas * np.array(C.classifier_regr_std).astype(dtype=np.float32)
		proposals[:,:4] = bbox_transform_inv(A, bbox_deltas)
		proposals = clip_boxes(proposals, [C.random_crop[0], C.random_crop[1]])
		if len(ignoreareas) > 0:
			ignore_overlap = box_op(np.ascontiguousarray(proposals[:, :4], dtype=np.float),
									np.ascontiguousarray(ignoreareas, dtype=np.float))
			ignore_sum = np.sum(ignore_overlap, axis=1)
			proposals[ignore_sum > igthre, -1] = 0
		keep = filter_negboxes(proposals, roi_stride)
		proposals[keep, -1] = 0
		valid_idxs = np.where(proposals[:, -1] == 1)[0]
		# initialise empty output objectives
		y_alf_overlap = np.zeros((all_anchors.shape[0], 1))
		y_alf_negindex = np.zeros((all_anchors.shape[0], 1))
		y_is_box_valid = np.zeros((all_anchors.shape[0], 1))
		y_alf_regr = np.zeros((all_anchors.shape[0], 4))

		valid_anchors = proposals[valid_idxs, :]
		valid_alf_overlap = np.zeros((valid_anchors.shape[0], 1))
		valid_is_box_valid = np.zeros((valid_anchors.shape[0], 1))
		valid_alf_regr = np.zeros((valid_anchors.shape[0], 4))
		if num_bboxes > 0:
			valid_overlap = bbox_overlaps(np.ascontiguousarray(valid_anchors, dtype=np.float),
										  np.ascontiguousarray(gta, dtype=np.float))
			# find every anchor close to which bbox
			argmax_overlaps = valid_overlap.argmax(axis=1)
			max_overlaps = valid_overlap[np.arange(len(valid_idxs)), argmax_overlaps]
			# find which anchor closest to every bbox
			gt_argmax_overlaps = valid_overlap.argmax(axis=0)
			gt_max_overlaps = valid_overlap[gt_argmax_overlaps, np.arange(num_bboxes)]
			gt_argmax_overlaps = np.where(valid_overlap == gt_max_overlaps)[0]
			valid_alf_overlap[gt_argmax_overlaps] = 1
			valid_alf_overlap[max_overlaps >= posthre] = 1
			for j in range(len(gta)):
				inds = valid_overlap[:, j].ravel().argsort()[-3:]
				valid_alf_overlap[inds] = 1
			# get positives labels
			fg_inds = np.where(valid_alf_overlap == 1)[0]
			valid_is_box_valid[fg_inds] = 1
			anchor_box = valid_anchors[fg_inds, :4]
			gt_box = gta[argmax_overlaps[fg_inds], :]

			# compute regression targets
			valid_alf_regr[fg_inds, :] = compute_targets(anchor_box, gt_box, C.classifier_regr_std, std=True)
			# get negatives labels
			bg_inds = np.where((max_overlaps < negthre) & (valid_is_box_valid.reshape((-1)) == 0))[0]
			valid_is_box_valid[bg_inds] = 1
			# transform to the original overlap and validbox
			y_alf_overlap[valid_idxs, :] = valid_alf_overlap
			y_is_box_valid[valid_idxs, :] = valid_is_box_valid
			y_alf_regr[valid_idxs, :] = valid_alf_regr
			y_alf_negindex = y_is_box_valid - y_alf_overlap
		y_alf_cls = np.expand_dims(np.concatenate([y_alf_overlap, y_alf_negindex], axis=1), axis=0)
		y_alf_regr = np.expand_dims(np.concatenate([y_alf_overlap, y_alf_regr], axis=1), axis=0)

		y_cls_batch.append(y_alf_cls)
		y_regr_batch.append(y_alf_regr)
	y_cls_batch = np.concatenate(np.array(y_cls_batch), axis=0)
	y_regr_batch = np.concatenate(np.array(y_regr_batch), axis=0)

	return [y_cls_batch, y_regr_batch]

def get_target_1st_posfirst(all_anchors, regr_layer, img_data, C,igthre=0.5,posthre=0.7,negthre=0.5):
	A = np.copy(all_anchors[:,:4])
	y_cls_batch, y_regr_batch = [], []
	for i in range(regr_layer.shape[0]):
		gta = np.copy(img_data[i]['bboxes'])
		num_bboxes = len(gta)
		ignoreareas = img_data[i]['ignoreareas']
		proposals = np.ones_like(all_anchors)
		bbox_deltas = regr_layer[i,:,:]
		bbox_deltas = bbox_deltas * np.array(C.classifier_regr_std).astype(dtype=np.float32)
		proposals[:,:4] = bbox_transform_inv(A, bbox_deltas)
		proposals = clip_boxes(proposals, [C.random_crop[0], C.random_crop[1]])

		# initialise empty output objectives
		y_alf_overlap = np.zeros((all_anchors.shape[0], 1))
		y_alf_negindex = np.zeros((all_anchors.shape[0], 1))
		y_is_box_valid = np.ones((all_anchors.shape[0], 1))
		y_alf_regr = np.zeros((all_anchors.shape[0], 4))

		if num_bboxes > 0:
			valid_overlap = bbox_overlaps(np.ascontiguousarray(proposals, dtype=np.float),
										  np.ascontiguousarray(gta, dtype=np.float))
			# find every anchor close to which bbox
			argmax_overlaps = valid_overlap.argmax(axis=1)
			max_overlaps = valid_overlap[np.arange(len(proposals)), argmax_overlaps]
			# find which anchor closest to every bbox
			gt_argmax_overlaps = valid_overlap.argmax(axis=0)
			gt_max_overlaps = valid_overlap[gt_argmax_overlaps, np.arange(num_bboxes)]
			gt_argmax_overlaps = np.where(valid_overlap == gt_max_overlaps)[0]
			y_alf_overlap[gt_argmax_overlaps] = 1
			y_alf_overlap[max_overlaps >= posthre] = 1
			for j in range(len(gta)):
				inds = valid_overlap[:, j].ravel().argsort()[-3:]
				y_alf_overlap[inds] = 1
			# get positives labels
			fg_inds = np.where(y_alf_overlap == 1)[0]
			anchor_box = proposals[fg_inds, :4]
			gt_box = gta[argmax_overlaps[fg_inds], :]

			# compute regression targets
			y_alf_regr[fg_inds, :] = compute_targets(anchor_box, gt_box, C.classifier_regr_std, std=True)
			# get negatives labels
			bg_inds = np.where(max_overlaps < negthre)[0]
			y_alf_negindex[bg_inds] = 1

			# find the invalid anchors
			if len(ignoreareas) > 0:
				ignore_overlap = box_op(np.ascontiguousarray(proposals[:, :4], dtype=np.float),
										np.ascontiguousarray(ignoreareas, dtype=np.float))
				ignore_sum = np.sum(ignore_overlap, axis=1)
				y_is_box_valid[ignore_sum > igthre] = 0
				y_alf_negindex = (np.logical_and(y_alf_negindex, y_is_box_valid)).astype(np.float)
		else:
			y_alf_negindex = np.ones((all_anchors.shape[0], 1))
			if len(ignoreareas) > 0:
				ignore_overlap = box_op(np.ascontiguousarray(proposals[:, :4], dtype=np.float),
										np.ascontiguousarray(ignoreareas, dtype=np.float))
				ignore_sum = np.sum(ignore_overlap, axis=1)
				y_is_box_valid[ignore_sum > igthre] = 0
				y_alf_negindex = (np.logical_and(y_alf_negindex, y_is_box_valid)).astype(np.float)

		y_alf_cls = np.expand_dims(np.concatenate([y_alf_overlap, y_alf_negindex], axis=1), axis=0)
		y_alf_regr = np.expand_dims(np.concatenate([y_alf_overlap, y_alf_regr], axis=1), axis=0)
		y_cls_batch.append(y_alf_cls)
		y_regr_batch.append(y_alf_regr)
	y_cls_batch = np.concatenate(np.array(y_cls_batch), axis=0)
	y_regr_batch = np.concatenate(np.array(y_regr_batch), axis=0)
	return [y_cls_batch, y_regr_batch]


def get_target_2nd(anchors, regr_layer, img_data, C, roi_stride=10,igthre=0.5,posthre=0.7,negthre=0.5):
	y_cls_batch, y_regr_batch = [], []
	anc_len = anchors.shape[1]
	for i in range(regr_layer.shape[0]):
		gta = img_data[i]['bboxes']
		num_bboxes = len(gta)
		ignoreareas = img_data[i]['ignoreareas']
		proposals = np.copy(anchors[i,:,:])

		bbox_deltas = regr_layer[i,:,:]

		bbox_deltas = bbox_deltas * np.array(C.classifier_regr_std).astype(dtype=np.float32)
		proposals[:,:4] = bbox_transform_inv(proposals[:,:4], bbox_deltas)
		proposals = clip_boxes(proposals, [C.random_crop[0], C.random_crop[1]])
		if len(ignoreareas) > 0:
			ignore_overlap = box_op(np.ascontiguousarray(proposals[:, :4], dtype=np.float),
									np.ascontiguousarray(ignoreareas, dtype=np.float))
			ignore_sum = np.sum(ignore_overlap, axis=1)
			proposals[ignore_sum > igthre, -1] = 0
		keep = filter_negboxes(proposals, roi_stride)
		proposals[keep, -1] = 0
		valid_idxs = np.where(proposals[:, -1] == 1)[0]
		# initialise empty output objectives
		y_alf_overlap = np.zeros((anc_len, 1))
		y_alf_negindex = np.zeros((anc_len, 1))
		y_is_box_valid = np.zeros((anc_len, 1))
		y_alf_regr = np.zeros((anc_len, 4))

		valid_anchors = proposals[valid_idxs, :]
		valid_alf_overlap = np.zeros((valid_anchors.shape[0], 1))
		valid_is_box_valid = np.zeros((valid_anchors.shape[0], 1))
		valid_alf_regr = np.zeros((valid_anchors.shape[0], 4))
		if num_bboxes > 0:
			valid_overlap = bbox_overlaps(np.ascontiguousarray(valid_anchors, dtype=np.float),
										  np.ascontiguousarray(gta, dtype=np.float))
			# find every anchor close to which bbox
			argmax_overlaps = valid_overlap.argmax(axis=1)
			max_overlaps = valid_overlap[np.arange(len(valid_idxs)), argmax_overlaps]
			# find which anchor closest to every bbox
			gt_argmax_overlaps = valid_overlap.argmax(axis=0)
			gt_max_overlaps = valid_overlap[gt_argmax_overlaps, np.arange(num_bboxes)]
			gt_argmax_overlaps = np.where(valid_overlap == gt_max_overlaps)[0]
			valid_alf_overlap[gt_argmax_overlaps] = 1
			valid_alf_overlap[max_overlaps >= posthre] = 1
			for j in range(len(gta)):
				inds = valid_overlap[:, j].ravel().argsort()[-3:]
				valid_alf_overlap[inds] = 1
			# get positives labels
			fg_inds = np.where(valid_alf_overlap == 1)[0]
			valid_is_box_valid[fg_inds] = 1
			anchor_box = valid_anchors[fg_inds, :4]
			gt_box = gta[argmax_overlaps[fg_inds], :]
			# compute regression targets
			valid_alf_regr[fg_inds, :] = compute_targets(anchor_box, gt_box, C.classifier_regr_std, std=True)
			# get negatives labels
			bg_inds = np.where((max_overlaps < negthre) & (valid_is_box_valid.reshape((-1)) == 0))[0]
			valid_is_box_valid[bg_inds] = 1
			# transform to the original overlap and validbox
			y_alf_overlap[valid_idxs, :] = valid_alf_overlap
			y_is_box_valid[valid_idxs, :] = valid_is_box_valid
			y_alf_regr[valid_idxs, :] = valid_alf_regr
			y_alf_negindex = y_is_box_valid - y_alf_overlap
		y_alf_cls = np.expand_dims(np.concatenate([y_alf_overlap, y_alf_negindex], axis=1), axis=0)
		y_alf_regr = np.expand_dims(np.concatenate([y_alf_overlap, y_alf_regr], axis=1), axis=0)

		y_cls_batch.append(y_alf_cls)
		y_regr_batch.append(y_alf_regr)
	y_cls_batch = np.concatenate(np.array(y_cls_batch), axis=0)
	y_regr_batch = np.concatenate(np.array(y_regr_batch), axis=0)

	return [y_cls_batch, y_regr_batch]
def generate_pp_2nd(all_anchors, regr_layer, C):
	A = np.copy(all_anchors[:,:4])
	proposals_batch = []
	for i in range(regr_layer.shape[0]):
		proposals = np.ones_like(all_anchors)

		bbox_deltas = regr_layer[i,:,:]

		bbox_deltas = bbox_deltas * np.array(C.classifier_regr_std).astype(dtype=np.float32)
		proposals[:,:4] = bbox_transform_inv(A, bbox_deltas)
		proposals = clip_boxes(proposals, [C.random_crop[0], C.random_crop[1]])
		proposals_batch.append(np.expand_dims(proposals,axis=0))
	return np.concatenate(proposals_batch, axis=0)

# for inference
def pred_pp_1st(anchors, cls_pred, regr_pred, C):
	A = np.copy(anchors[:, :4])
	scores = cls_pred[0, :, :]
	bbox_deltas = regr_pred.reshape((-1, 4))
	bbox_deltas = bbox_deltas * np.array(C.classifier_regr_std).astype(dtype=np.float32)
	proposals = bbox_transform_inv(A, bbox_deltas)
	proposals = clip_boxes(proposals, [C.random_crop[0], C.random_crop[1]])
	proposals = np.concatenate((proposals, scores), axis=-1)
	return proposals
def pred_pp_2nd(anchors, cls_pred, regr_pred, C):
	scores = cls_pred[0, :, :]
	bbox_deltas = regr_pred.reshape((-1, 4))
	bbox_deltas = bbox_deltas * np.array(C.classifier_regr_std).astype(dtype=np.float32)
	anchors[:, :4] = bbox_transform_inv(anchors[:, :4], bbox_deltas)
	anchors[:, :4] = clip_boxes(anchors[:, :4], [C.random_crop[0], C.random_crop[1]])
	proposals = np.concatenate((anchors, scores), axis=-1)
	return proposals
def pred_det(anchors, cls_pred, regr_pred, C, step=1):
	if step == 1:
		scores = cls_pred[0, :, :]
	elif step == 2:
		scores = anchors[:, -1:] * cls_pred[0, :, :]
	elif step == 3:
		scores = anchors[:, -2:-1] * anchors[:, -1:] * cls_pred[0, :, :]
	A = np.copy(anchors[:, :4])
	bbox_deltas = regr_pred.reshape((-1, 4))
	bbox_deltas = bbox_deltas * np.array(C.classifier_regr_std).astype(dtype=np.float32)

	proposals = bbox_transform_inv(A, bbox_deltas)
	proposals = clip_boxes(proposals, [C.random_crop[0], C.random_crop[1]])
	keep = filter_boxes(proposals, C.roi_stride)
	proposals = proposals[keep, :]
	scores = scores[keep]

	order = scores.ravel().argsort()[::-1]
	order = order[:C.pre_nms_topN]
	proposals = proposals[order, :]
	scores = scores[order]

	keep = np.where(scores > C.scorethre)[0]
	proposals = proposals[keep, :]
	scores = scores[keep]
	keep = nms(np.hstack((proposals, scores)), C.overlap_thresh, usegpu=False, gpu_id=0)

	keep = keep[:C.post_nms_topN]
	proposals = proposals[keep, :]
	scores = scores[keep]
	return proposals, scores
