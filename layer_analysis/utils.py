from keras.backend import image_data_format


def get_input_shape(height, width, channels = 3):
    if image_data_format() == 'channels_first':
        return (channels, height, width)
    else:
        return (height, width, channels)


class InvalidPatchSizeParameter(Exception):

    def __init__(self, message=
        "The patch width and/or patch height is too large compared to the size of the image."
        "\n\tAs a result, the matricies created do not have enough data for training."
    ):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'\n\n\t{self.message}'