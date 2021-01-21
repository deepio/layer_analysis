from __future__ import division

import keras
import cv2
import numpy as np

from keras.models import load_model
from keras.backend import image_data_format


def process_image(image, model_paths, w_height, w_width, mode='masks'):
    """
    Takes a document image and pre-trained SAE model paths
    and returns a single image with logical labels.
    """

    num_labels = len(model_paths)

    sae_models = []
    for id_label in range(num_labels):
        sae_models.append(load_model(model_paths[id_label]))

    [img_height, img_width, channels] = image.shape

    if mode == 'masks':
        output_images = []

        for id_label in range(num_labels):
            output_images.append(np.zeros((img_height, img_width)))

    elif mode == 'logical':
        output_image = np.zeros((img_height, img_width), 'uint8')

    for row in range(0, img_height - w_height - 1, w_height):
        print(str(row) + ' / ' + str(img_height))
        for col in range(0, img_width - w_width - 1, w_width):
            sample = image[row:row+w_height, col:col+w_width]

            # Pre-process (check that training does the same!)
            sample = (255. - sample) / 255.

            if image_data_format() == 'channels_first':
                sample = np.asarray(sample).reshape(1, 3, w_height, w_width)
            else:
                sample = np.asarray(sample).reshape(1, w_height, w_width, 3)

            if mode == 'masks':

                for id_label in range(num_labels):
                    prediction = sae_models[id_label].predict(sample)
                    output_images[id_label][row:row+w_height,col:col+w_width] = 100*prediction[0,:,:,0]

            elif mode == 'logical':
                predictions = []

                for id_label in range(num_labels):
                    predictions.append( sae_models[id_label].predict(sample)[0,:,:,0]  )

                output_image[row:row+w_height,col:col+w_width] = np.argmax( predictions, axis = 0 )

    if mode == 'masks':
        return output_images
    elif mode == 'logical':
        return output_image