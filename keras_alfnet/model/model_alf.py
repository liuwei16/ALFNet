from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from keras.layers import *
from keras import backend as K
import numpy as np
import math

def prior_probability(probability=0.01):
	def f(shape, dtype=K.floatx()):
		# set bias to -log((1 - p)/p) for foregound
		result = np.ones(shape, dtype=dtype) * -math.log((1 - probability) / probability)
		return result
	return f

def alf_pred(input,num_anchors,name,filters=256,kersize=(3,3),trainable=True):
    # the first layer modified from256 to 512
    x = Convolution2D(filters, kersize, padding='same', activation='relu',
                      kernel_initializer='glorot_normal', name=name + '_conv', trainable=trainable)(input)

    x_class = Convolution2D(num_anchors, (1, 1),activation='sigmoid',
                            kernel_initializer='glorot_normal',
                            bias_initializer=prior_probability(),
                            name=name+'_rpn_class',trainable=trainable)(x)
    x_class_reshape = Reshape((-1, 1), name=name+'_class_reshape')(x_class)

    x_regr = Convolution2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='glorot_normal',
                           name=name+'_rpn_regress',trainable=trainable)(x)
    x_regr_reshape = Reshape((-1,4), name=name+'_regress_reshape')(x_regr)
    return x_class_reshape, x_regr_reshape

def alf_1st(base_layers, num_anchors,filters=256,kersize=(3,3), trainable=True):
    P6 = Convolution2D(256, (3, 3), strides=2, padding='same', activation='relu', kernel_initializer='glorot_normal',
                       name='P6', trainable=trainable)(base_layers[2])
    P3 = Convolution2D(512, (3, 3), strides=1, padding='same', activation='relu', kernel_initializer='glorot_normal',
                       name='P3', trainable=trainable)(base_layers[0])

    P3_cls, P3_regr = alf_pred(P3, num_anchors[0], name='pred0_1', filters=filters, kersize=kersize, trainable=trainable)
    P4_cls, P4_regr = alf_pred(base_layers[1], num_anchors[1], name='pred1_1', filters=filters, kersize=kersize, trainable=trainable)
    P5_cls, P5_regr = alf_pred(base_layers[2], num_anchors[2], name='pred2_1', filters=filters, kersize=kersize, trainable=trainable)
    P6_cls, P6_regr = alf_pred(P6, num_anchors[3], name='pred3_1', filters=filters, kersize=kersize, trainable=trainable)
    y_cls = Concatenate(axis=1, name='mbox_cls_1')([P3_cls, P4_cls, P5_cls, P6_cls])
    y_regr = Concatenate(axis=1, name='mbox_regr_1')([P3_regr, P4_regr, P5_regr, P6_regr])
    return [y_cls, y_regr, P3, base_layers[1], base_layers[2], P6]

def alf_2nd(P3, P4, P5, P6, num_anchors, filters=256, kersize=(3,3),trainable=True):
    P3_cls, P3_regr = alf_pred(P3, num_anchors[0], name='pred0_2', filters=filters, kersize=kersize, trainable=trainable)
    P4_cls, P4_regr = alf_pred(P4, num_anchors[1], name='pred1_2', filters=filters, kersize=kersize, trainable=trainable)
    P5_cls, P5_regr = alf_pred(P5, num_anchors[2], name='pred2_2', filters=filters, kersize=kersize, trainable=trainable)
    P6_cls, P6_regr = alf_pred(P6, num_anchors[3], name='pred3_2', filters=filters, kersize=kersize, trainable=trainable)

    y_cls = Concatenate(axis=1, name='mbox_cls_2')([P3_cls, P4_cls, P5_cls, P6_cls])
    y_regr = Concatenate(axis=1, name='mbox_regr_2')([P3_regr, P4_regr, P5_regr, P6_regr])

    return [y_cls, y_regr]

def alf_3rd(P3, P4, P5, P6, num_anchors, filters=256, kersize=(3,3),trainable=True):
    P3_cls, P3_regr = alf_pred(P3, num_anchors[0], name='pred0_3', filters=filters, kersize=kersize, trainable=trainable)
    P4_cls, P4_regr = alf_pred(P4, num_anchors[1], name='pred1_3', filters=filters, kersize=kersize, trainable=trainable)
    P5_cls, P5_regr = alf_pred(P5, num_anchors[2], name='pred2_3', filters=filters, kersize=kersize, trainable=trainable)
    P6_cls, P6_regr = alf_pred(P6, num_anchors[3], name='pred3_3', filters=filters, kersize=kersize, trainable=trainable)

    y_cls = Concatenate(axis=1, name='mbox_cls_3')([P3_cls, P4_cls, P5_cls, P6_cls])
    y_regr = Concatenate(axis=1, name='mbox_regr_3')([P3_regr, P4_regr, P5_regr, P6_regr])

    return [y_cls, y_regr]

def create_alf(base_layers, num_anchors,trainable=True,steps=3):
    alf1 = alf_1st(base_layers, num_anchors, trainable=trainable)
    if steps==1:
        return alf1[:2]
    elif steps==2:
        alf2 = alf_2nd(alf1[2], alf1[3], alf1[4], alf1[5], num_anchors, trainable=trainable)
        return alf1[:2], alf2
    elif steps==3:
        alf2 = alf_2nd(alf1[2], alf1[3], alf1[4], alf1[5], num_anchors, trainable=trainable)
        alf3 = alf_3rd(alf1[2], alf1[3], alf1[4], alf1[5], num_anchors, trainable=trainable)
        return alf1[:2], alf2, alf3