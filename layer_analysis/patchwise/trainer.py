from __future__ import division

import cv2
import numpy as np
import random as rd
from keras.models import Model
from keras.layers import Dropout, UpSampling2D, Concatenate
from keras.layers import Conv2D, MaxPooling2D, Input
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.backend import image_data_format
import keras
import tensorflow as tf

import layer_analysis as la
from memory_profiler import profile


def get_sae(height, width, pretrained_weights = None):
    ff = 32

    inputs = Input(shape=la.utils.get_input_shape(height,width))
    conv1 = Conv2D(filters=ff, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(ff, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(ff * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(ff * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(ff * 8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv7 = Conv2D(ff * 8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    up8 = UpSampling2D(size=(2, 2))(conv7)
    up8 = Conv2D(ff * 4, 2, activation='relu', padding='same', kernel_initializer='he_normal')(up8)
    merge8 = Concatenate(axis = 3)([conv2,up8])

    conv8 = Conv2D(ff * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(ff * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    up9 = UpSampling2D(size=(2, 2))(conv8)
    up9 = Conv2D(ff * 2, 2, activation='relu', padding='same', kernel_initializer='he_normal')(up9)
    merge9 = Concatenate(axis = 3)([conv1,up9])

    conv9 = Conv2D(ff * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(ff * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    if pretrained_weights is not None:
        model.load_weights(pretrained_weights)

    return model

fp=open(f"memory_profiler.get_train.{la.__version__}.log", "a")
@profile(stream=fp)
def getTrain(input_image, gt, patch_height, patch_width, max_samples_per_class=la._SAMPLES_PER_CLASS_, factor=la._SPEED_FACTOR_):
    # factor = 100. 

    X_train = {}
    Y_train = {}

    ratio = {}
    count = {}
    X_train = None
    Y_train = None

    count = (gt == 1).sum()
    samples_per_class = min(count, max_samples_per_class)
    ratio = factor * (samples_per_class/float(count))
    

    # Get samples according to the ratio per label
    height, width, _ = input_image.shape

    # pre-process entire image.
    # 255 turns into 0, 0 turns to 1. All ints are between 0.0 and 1.0.
    input_image = (255. - input_image) / 255.

    for row in range(patch_height, height-patch_height-1):
        for col in range(patch_width, width-patch_width-1):

            # get 1 for every factor-ish
            if rd.random() < 1./factor:
  
                if gt[row][col] == 1:

                    if rd.random() < ratio: # Take samples according to its ratio

                        from_x = row-(patch_height//2)
                        from_y = col-(patch_width//2)

                        sample_x = input_image[from_x:from_x+patch_height,from_y:from_y+patch_width]
                        sample_y = np.expand_dims(gt[from_x:from_x+patch_height,from_y:from_y+patch_width], axis=-1)

                        # Empty in the first iteration
                        if X_train is None:
                          X_train = np.asarray([sample_x])
                          Y_train = np.asarray([sample_y])
                        # Combine without flattening in every other iteration
                        else:
                          X_train = np.concatenate((X_train, [sample_x]))
                          Y_train = np.concatenate((Y_train, [sample_y]))

                        yield X_train, Y_train


fp=open(f"memory_profiler.train_msae.{la.__version__}.log", "a")
@profile(stream=fp)
def train_msae(input_image, gt, patch_height, patch_width, output_path, epochs=la._EPOCHS_, max_samples_per_class=la._SAMPLES_PER_CLASS_, batch_size=la._BATCH_SIZE_, validation_split=la._VALIDATION_SPLIT_):

    height, width, _ = input_image.shape
    # total_patches_estimation = (height - (patch_height*2)) * (width - (patch_width*2)) / la._SPEED_FACTOR_
    total_patches_estimation = 50

    # Training loop
    for label in gt:

        print('Training created with:')
        print('Training a new model for ' + str(label))
        model = get_sae(
            height=patch_height,
            width=patch_width,
        )

        model.summary()
        callbacks_list = [
            ModelCheckpoint(output_path[label], save_best_only=True, monitor='val_accuracy', verbose=1, mode='max'),
            EarlyStopping(monitor='val_accuracy', patience=3, verbose=0, mode='max')
        ]

        model.fit_generator(
            generator=getTrain(input_image, gt[label], patch_height, patch_width, max_samples_per_class),
            verbose=2,
            callbacks=callbacks_list,
            epochs=epochs,
            steps_per_epoch=total_patches_estimation,
        )