#!/usr/bin/env python

from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import nibabel
import os
import sys
import matplotlib.pyplot as plt
import skimage
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import ZeroPadding3D, Conv3D, MaxPooling3D, BatchNormalization
import h5py

#if not tf.executing_eagerly():
#    tf.enable_eager_execution()
#    print("executing eagerly")

print("loading shapes.h5")
f = h5py.File('/home/hayashis/data/shapes.h5', 'r')
x = f['x'][:]
y = f['y'][:]
class_names = f['class_names'][:]
input_shape = f['input_shape'][:]
    
#split input into training and test
#TODO - I should randomize this
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1,random_state=0)
print(["x_test", x_test.shape, "y_test", y_test.shape])
print(["class_names", class_names.shape]);
print(["input_shape", input_shape.shape]);

print("loading model")
model = tf.keras.models.load_model('fitmodel.h5')
model.summary()

#print("evaluating")
#score = model.evaluate(x_test, y_test)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])

print("predicting x_test")
y_pred = model.predict(x_test)
print(y_test.shape)
print(np.argmax(y_test, axis=1).shape)

#from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
print(y_test.shape);
print(y_pred.shape);
print(cm.shape);
#for i in range(0, len(class_names)):
#	print(i, class_names[i], cm[i][i])
plt.figure()
plt.matshow(cm)
plt.show()


