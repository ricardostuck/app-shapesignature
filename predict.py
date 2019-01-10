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

print("loading model")
model = tf.keras.models.load_model('fitmodel.h5')
model.summary()

print("loading testdata")
img = nibabel.load("testdata/113922/masks/CC_5.nii.gz")
data = img.get_fdata()

print("prepping data")
bounds = np.sort(np.vstack(np.nonzero(data)))[:, [0, -1]]
x, y, z = bounds
cropped_data = data[x[0]:x[1], y[0]:y[1], z[0]:z[1]]
input_shape = (64,64,64,1)
resized_data = skimage.transform.resize(cropped_data, input_shape, anti_aliasing=True)
print(resized_data.shape)

#create x
x = np.array([resized_data])
class_names = os.listdir("testdata/113922/masks")

#run predict
y = model.predict(x)
for i in range(0, len(y[0])):
	score = y[0][i]
	if score < 0.001:
		score = 0
	print(score, class_names[i])

#from keras import backend as K
#def get_activations(model, layer, X_batch):
#    get_activations = K.function([model.layers[0].input, K.learning_phase()], model.layers[layer].output)
#    return get_activations([X_batch,0])

int_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer('flatten').output)
y = int_model.predict(x)
print(y[0])

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

