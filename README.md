# Shape Signature

## Output

This App will take Brainlife's neuro/tractmasks as input and generate `signature.json` with soemthing like the following content

```
{
    "UF_left.nii.gz": [-0.4391258656978607, -0.4570843279361725, ... -0.11965778470039368, -0.38888874650001526],   
    ...
    "T_POSTC_right.nii.gz": [-0.4391258656978607, -0.4570843279361725, ... -0.11965778470039368, -0.38888874650001526],
}

```

Each tract will have values equals to the size of flatten layer output which is currently set to 2048.

## Author

Soichi Hayashi <hayashis@iu.edu>

## Classification Model

The classification model was trained using TractSeg outputs from roughtly 100 subjects (20% of them used to validate the model). The model can accurately classify each tracts at near 100% accuracy, but we discard the layer below the flatten layer as we are not interested in classifiying tracts - we use the trained model to extract activation outputs from the conv3d layers.

Layer (type)                 Output Shape              Param #   
=================================================================
conv3d (Conv3D)              (None, 60, 60, 60, 8)     1008      
_________________________________________________________________
conv3d_1 (Conv3D)            (None, 56, 56, 56, 16)    16016     
_________________________________________________________________
max_pooling3d (MaxPooling3D) (None, 14, 14, 14, 16)    0         
_________________________________________________________________
dropout (Dropout)            (None, 14, 14, 14, 16)    0         
_________________________________________________________________
conv3d_2 (Conv3D)            (None, 11, 11, 11, 16)    16400     
_________________________________________________________________
conv3d_3 (Conv3D)            (None, 9, 9, 9, 32)       13856     
_________________________________________________________________
max_pooling3d_1 (MaxPooling3 (None, 4, 4, 4, 32)       0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 4, 4, 4, 32)       0         
_________________________________________________________________
batch_normalization (BatchNo (None, 4, 4, 4, 32)       128       
_________________________________________________________________
flatten (Flatten)            (None, 2048)              0         
_________________________________________________________________
dense (Dense)                (None, 144)               295056    
_________________________________________________________________
dropout_2 (Dropout)          (None, 144)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 72)                10440     
=================================================================
Total params: 352,904
Trainable params: 352,840
Non-trainable params: 64

```
5235/5235 [==============================] - 71s 14ms/step - loss: 0.0098 - acc: 0.3727 - val_loss: 0.0045 - val_acc: 0.7815
Epoch 2/8
5235/5235 [==============================] - 64s 12ms/step - loss: 0.0022 - acc: 0.8877 - val_loss: 6.6277e-04 - val_acc: 0.9672
Epoch 3/8
5235/5235 [==============================] - 64s 12ms/step - loss: 5.4972e-04 - acc: 0.9759 - val_loss: 1.4308e-04 - val_acc: 0.9931
Epoch 4/8
5235/5235 [==============================] - 64s 12ms/step - loss: 2.3532e-04 - acc: 0.9901 - val_loss: 1.1028e-04 - val_acc: 0.9947
Epoch 5/8
5235/5235 [==============================] - 64s 12ms/step - loss: 1.9655e-04 - acc: 0.9914 - val_loss: 9.6726e-05 - val_acc: 0.9954
Epoch 6/8
5235/5235 [==============================] - 64s 12ms/step - loss: 1.4950e-04 - acc: 0.9943 - val_loss: 1.1865e-04 - val_acc: 0.9954
Epoch 7/8
5235/5235 [==============================] - 65s 12ms/step - loss: 9.4822e-05 - acc: 0.9966 - val_loss: 5.9258e-05 - val_acc: 0.9969
Epoch 8/8
5235/5235 [==============================] - 65s 12ms/step - loss: 1.0702e-04 - acc: 0.9952 - val_loss: 8.0987e-05 - val_acc: 0.9962
evaluating
728/728 [==============================] - 5s 7ms/step
Test loss: 7.794678107513892e-05
Test accuracy: 0.9958791208791209

```
