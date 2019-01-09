#!/usr/bin/env python

from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import nibabel
import os
import sys
import matplotlib.pyplot as plt
import matplotlib
import skimage
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import ZeroPadding3D, Conv3D, MaxPooling3D, BatchNormalization
import h5py

model = tf.keras.models.load_model('fitmodel.h5')
model.summary()

#print(model.get_config())
#print(model.layers[0].get_config())
#print(model.count_params())
#print(model.layers[1].count_params())

#print(model.layers[1].weights) #(4, 4, 4, 8, 16)
#print(model.layers[1].get_weights())

#def plot_filters(layer, x, y):
#	weights = layer.get_weights()[0] #[1] is for bias
#	weights = np.rollaxis(weights,4,0) #bring filter index to the front
#	#print(weights.shape)
#	fig = plt.figure()
#	for j in range(len(weights)):
#		print(j)
#		ax = fig.add_subplot(y,x,j+1)
#		#print(filters[:][:][:][j])
#		#print(weights[j][0].shape)
#		#img = np.reshape(weights[j][0], (4,4))
#		ax.matshow(weights[j][0][:][:][0], cmap = matplotlib.cm.binary)
#		plt.xticks(np.array([]))
#		plt.yticks(np.array([]))
#	plt.tight_layout()
#	plt.show()
#
#plot_filters(model.layers[1],4,4)

