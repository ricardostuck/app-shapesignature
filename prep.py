#!/usr/bin/env python

import numpy as np
import nibabel
import os
import sys
import skimage
import h5py
import tensorflow as tf

class_names = os.listdir("/home/hayashis/data/shapes/5bc50099d785d7004357d360/masks")
ids = os.listdir("/home/hayashis/data/shapes")

#slices to show for quick check
images = []
images_class = []
input_shape = (64,64,64,1)

for id in ids:
    #only load if it looks like mongo id
    if len(id) != 24:
        continue	

    #load 200 subjects
    if len(images) > len(class_names)*200:
        print("loaded enough input")
        break 

    for file in os.listdir("/home/hayashis/data/shapes/"+id+"/masks"):

        print("reading", len(images), id, file)
        img = nibabel.load("/home/hayashis/data/shapes/"+id+"/masks/"+file)
        data = img.get_fdata()

        #TractSeg seems to generate near identical tractmask for OR and T_OCC. Let's cheat and 
        #label OR as T_OCC..#TractSeg seems to generate near identical tractmask for OR and T_OCC. Let's cheat and 
        #label OR as T_OCC..
        if file == "OR_left.nii.gz":
            file = "T_OCC_left.nii.gz"
        if file == "OR_right.nii.gz":
            file="T_OCC_right.nii.gz"
        
        #find binding box that contains mask and crop
        bounds = np.sort(np.vstack(np.nonzero(data)))[:, [0, -1]]
        x, y, z = bounds
        cropped_data = data[x[0]:x[1], y[0]:y[1], z[0]:z[1]]
        #print(cropped_data.shape)

        #resize to fit 64x64x64 box.
        #TODO - I should resize it while retaining the overall shape. resize
        #would just stretch/shrink to fit the box in all axes
        resized_data = skimage.transform.resize(cropped_data, input_shape, anti_aliasing=True)
        
        #TODO - normalize?
        print(resized_data.shape)
        
        #sample display
        #if file == "CA.nii.gz":
        #    plt.figure()
        #    plt.imshow(resized_data[20, :, :, 0])
        #    plt.show()
            
        images.append(resized_data)
        images_class.append(class_names.index(file))
            
x = np.array(images)
y = tf.keras.utils.to_categorical(images_class, len(class_names))

with h5py.File('/home/hayashis/data/shapes.h5', 'w') as f:
 	f.create_dataset('x', data=x)
 	f.create_dataset('y', data=y)
 	f.create_dataset('class_names', data=class_names)
 	f.create_dataset('input_shape', data=input_shape)
 	#f.create_dataset('classes', images_classes, dtype='s')

