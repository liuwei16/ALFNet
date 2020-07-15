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

def PredHead_RFB(input,input_2,num_anchors,name,filters=256,kersize=(3,3),trainable=True):
    # the first layer modified from256 to 512
    x = Convolution2D(filters, kersize, padding='same', activation='relu',
                      kernel_initializer='glorot_normal', name=name + '_conv', trainable=trainable)(input)
                      
    x_mix = Add(name=name + '_add')([x, input_2])

    x_class = Convolution2D(num_anchors, (1, 1),activation='sigmoid',
                            kernel_initializer='glorot_normal',
                            bias_initializer=prior_probability(),
                            name=name+'_rpn_class',trainable=trainable)(x_mix)
    x_class_reshape = Reshape((-1, 1), name=name+'_class_reshape')(x_class)

    x_regr = Convolution2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='glorot_normal',
                           name=name+'_rpn_regress',trainable=trainable)(x_mix)
    x_regr_reshape = Reshape((-1,4), name=name+'_regress_reshape')(x_regr)
    return x_class_reshape, x_regr_reshape
    
def PredHead(input,num_anchors,name,filters=256,kersize=(3,3),trainable=True):
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

#Visible-part Estimation
def VE(base_layers, num_anchors,filters=256,kersize=(3,3), trainable=True):
    P6 = Convolution2D(256, (3, 3), strides=2, padding='same', activation='relu', kernel_initializer='glorot_normal',
                       name='P6', trainable=trainable)(base_layers[2])
    P3 = Convolution2D(512, (3, 3), strides=1, padding='same', activation='relu', kernel_initializer='glorot_normal',
                       name='P3', trainable=trainable)(base_layers[0])

    P3_cls, P3_regr = PredHead(P3, num_anchors[0], name='pred0_1', filters=filters, kersize=kersize, trainable=trainable)
    P4_cls, P4_regr = PredHead(base_layers[1], num_anchors[1], name='pred1_1', filters=filters, kersize=kersize, trainable=trainable)
    P5_cls, P5_regr = PredHead(base_layers[2], num_anchors[2], name='pred2_1', filters=filters, kersize=kersize, trainable=trainable)
    P6_cls, P6_regr = PredHead(P6, num_anchors[3], name='pred3_1', filters=filters, kersize=kersize, trainable=trainable)
    y_cls = Concatenate(axis=1, name='mbox_cls_1')([P3_cls, P4_cls, P5_cls, P6_cls])
    y_regr = Concatenate(axis=1, name='mbox_regr_1')([P3_regr, P4_regr, P5_regr, P6_regr])
    return [y_cls, y_regr, P3, base_layers[1], base_layers[2], P6]

#Full-body Refinement
def FR(P3, P4, P5, P6, num_anchors, filters=256, kersize=(3,3),trainable=True):
    P6_upsampled = UpSampling2D(size=(2, 2), name='P6_upsampled')(P6)
    P6_upsampled_2=Convolution2D(256, (3, 3), strides=1, padding='same',activation='relu',kernel_initializer='glorot_normal',name='P6_upsampled_conv',trainable=trainable)(P6_upsampled)
    P5_upsampled = UpSampling2D(size=(2, 2), name='P5_upsampled')(P5)
    P5_upsampled_2=Convolution2D(256, (3, 3), strides=1, padding='same',activation='relu',kernel_initializer='glorot_normal',name='P5_upsampled_conv',trainable=trainable)(P5_upsampled)
    P4_upsampled = UpSampling2D(size=(2, 2), name='P4_upsampled')(P4)
    P4_upsampled_2=Convolution2D(256, (3, 3), strides=1, padding='same',activation='relu',kernel_initializer='glorot_normal',name='P4_upsampled_conv',trainable=trainable)(P4_upsampled)
    
    P3_cls, P3_regr = PredHead_RFB(P3,P4_upsampled_2,num_anchors[0], name='pred0_2', filters=filters, kersize=kersize, trainable=trainable)
    P4_cls, P4_regr = PredHead_RFB(P4,P5_upsampled_2,num_anchors[1], name='pred1_2', filters=filters, kersize=kersize, trainable=trainable)
    P5_cls, P5_regr = PredHead_RFB(P5,P6_upsampled_2,num_anchors[2], name='pred2_2', filters=filters, kersize=kersize, trainable=trainable)
    P6_cls, P6_regr = PredHead(P6,num_anchors[3], name='pred3_2', filters=filters, kersize=kersize, trainable=trainable)

    y_cls = Concatenate(axis=1, name='mbox_cls_2')([P3_cls, P4_cls, P5_cls, P6_cls])
    y_regr = Concatenate(axis=1, name='mbox_regr_2')([P3_regr, P4_regr, P5_regr, P6_regr])

    return [y_cls, y_regr]

def create_PRNet(base_layers, num_anchors,trainable=True,steps=3):
    ve = VE(base_layers, num_anchors, trainable=trainable)
    fr = FR(ve[2], ve[3], ve[4], ve[5], num_anchors, trainable=trainable)
    return ve[:2], fr
    
