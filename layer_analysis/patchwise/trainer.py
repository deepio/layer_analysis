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


def getTrain(input_image, gt, patch_height, patch_width, max_samples_per_class=la._SAMPLES_PER_CLASS_, factor=la._SPEED_FACTOR_):
    # factor = 100. 

    X_train = {}
    Y_train = {}

    # Calculate the ratio per label
    ratio = {}
    count = {}

    for label in gt:
        # Initialize data lists
        X_train[label] = []
        Y_train[label] = []

        count[label] = (gt[label] == 1).sum()
        samples_per_class = min(count[label], max_samples_per_class)
        ratio[label] = factor * (samples_per_class/float(count[label]))

    # Get samples according to the ratio per label
    height, width, _ = input_image.shape
    # compare the height/width of image with patch size.

    # pre-process entire image.
    # 255 turns into 0, 0 turns to 1. All ints are between 0.0 and 1.0.
    input_image = (255. - input_image) / 255.

    for row in range(patch_height, height-patch_height-1):
        for col in range(patch_width, width-patch_width-1):

            # get 1 for every factor-ish
            if rd.random() < 1./factor:

                for label in gt:
                    if gt[label][row][col] == 1:    


                        if rd.random() < ratio[label]: # Take samples according to its ratio
                            from_x = row-(patch_height//2)
                            from_y = col-(patch_width//2)

                            sample_x = input_image[from_x:from_x+patch_height,from_y:from_y+patch_width]
                            sample_y = gt[label][from_x:from_x+patch_height,from_y:from_y+patch_width]

                            X_train[label].append(sample_x)
                            Y_train[label].append(sample_y)

    # Manage different ordering 
    for label in gt:
        X_train[label] = np.asarray(X_train[label]).reshape(len(X_train[label]), patch_height, patch_width, 3)
        Y_train[label] = np.expand_dims(np.asarray(Y_train[label]), axis=-1)

    return [X_train, Y_train]



def train_msae(input_image, gt, height, width, output_path, epochs=la._EPOCHS_, max_samples_per_class=la._SAMPLES_PER_CLASS_, batch_size=la._BATCH_SIZE_, validation_split=la._VALIDATION_SPLIT_):

    # Create ground_truth
    [X_train, Y_train] = getTrain(input_image, gt, height, width, max_samples_per_class)

    # Training loop
    for label in Y_train:
        print('Training created with:')
        print("\t{} samples of {}".format(len(Y_train[label]),label))
        print('Training a new model for ' + str(label))
        model = get_sae(
            height=height,
            width=width
        )

        model.summary()
        callbacks_list = [
            ModelCheckpoint(output_path[label], save_best_only=True, monitor='val_accuracy', verbose=1, mode='max'),
            EarlyStopping(monitor='val_accuracy', patience=3, verbose=0, mode='max')
        ]

        # Training stage
        try:
            model.fit(
                x=X_train[label],
                y=Y_train[label],
                verbose=2,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=callbacks_list,
                epochs=epochs
            )
        except ValueError:
            raise la.utils.InvalidPatchSizeParameter from None
