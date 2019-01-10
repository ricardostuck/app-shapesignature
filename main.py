#!/usr/bin/env python

from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import ZeroPadding3D, Conv3D, MaxPooling3D, BatchNormalization
import numpy as np
import nibabel
import os
import skimage
import h5py
import json

print("loading model")
model = tf.keras.models.load_model('fitmodel.h5')
model.summary()

with open("config.json") as config_json:
    config = json.load(config_json)

    print("loading masks")
    images = []
    images_class = []
    input_shape = (64,64,64,1)
    for file in os.listdir(config["masks"]):
            img = nibabel.load(config["masks"]+"/"+file)
            data = img.get_fdata()

            #fit to input shape
            bounds = np.sort(np.vstack(np.nonzero(data)))[:, [0, -1]]
            x, y, z = bounds
            cropped_data = data[x[0]:x[1], y[0]:y[1], z[0]:z[1]]
            #print(cropped_data.shape)

            #resize to fit 64x64x64 box.
            #TODO - I should resize it while retaining the overall shape. resize
            #would just stretch/shrink to fit the box in all axes
            resized_data = skimage.transform.resize(cropped_data, input_shape, anti_aliasing=True)
            images.append(resized_data)
            images_class.append(file)

    #predict using model all the way to flatten
    print("predicting")
    int_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer('flatten').output)
    x = np.array(images)
    y = int_model.predict(x)


    #store to json
    print("outputting signature.json")
    out = {}
    for i in range(0, len(y)):
        out[images_class[i]] = y[i].tolist()
    with open("signature.json", "w") as f:
        json.dump(out, f)

