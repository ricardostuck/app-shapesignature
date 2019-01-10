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
#split=int(len(x)*0.2)
#x_test = x[0:split]
#y_test = y[0:split]
#x_train = x[split:]
#y_train = y[split:]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1,random_state=0)

print(["x_train", x_train.shape, "y_train", y_train.shape])
print(["x_test", x_test.shape, "y_test", y_test.shape])
print(["class_name.shape",class_names.shape])

class_size = len(class_names)
    
#build model
model = tf.keras.models.Sequential()

model.add(Conv3D(8,kernel_size=5,activation='relu',input_shape=input_shape))
model.add(Conv3D(16,kernel_size=5,activation='relu'))
model.add(MaxPooling3D(pool_size=4))
model.add(Dropout(0.3))

model.add(Conv3D(16,kernel_size=4,activation='relu'))
model.add(Conv3D(32,kernel_size=3,activation='relu'))
model.add(MaxPooling3D(pool_size=2))
model.add(Dropout(0.3))

model.add(BatchNormalization())
model.add(Flatten())

#model.add(Dense(class_size*4, activation='relu', name='dense1'))
#model.add(Dropout(0.4))
model.add(Dense(class_size*2, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(len(class_names), activation='softmax'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.summary()

print("fitting")
model.fit(x_train, y_train, batch_size=64, epochs=8, verbose=1, validation_split=0.2)
model.save('fitmodel.h5')

print("evaluating")
score = model.evaluate(x_test, y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


