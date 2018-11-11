import tensorflow as tf
from keras.engine import Layer
from tensorflow import Tensor

from src.util.util import softmax_temperature


class DistToLab(Layer):
    """
    Layer to convert color distribution into lab color space
    """

    def __init__(self, color_map: Tensor, **kwargs):
        """
        Create layer to convert distribution into lab color space
        :param shape: Shape of grayscale input
        :param kwargs:
        """
        super(DistToLab, self).__init__(**kwargs)
        self.color_map = color_map

    def call(self, x, mask=None):
        [grayscale, color_classes] = x

        # Flatten classes into 2D array
        color_classes_flat = tf.reshape(color_classes, (-1, tf.shape(color_classes)[-1]))
        # Apply softmax with low temperature to create approximate one hot encoding
        color_classes_flat = softmax_temperature(color_classes_flat)

        # Use matrix multiplication to lookup color and sum for each probability
        ab_colors = tf.matmul(color_classes_flat, self.color_map)
        # Reshape ab colors into 2D image plus channels
        ab_colors = tf.reshape(ab_colors, (-1, 256, 256, 2))
        # Append grayscale channel
        lab_colors = tf.concat([grayscale, ab_colors], axis=-1)

        return lab_colors

    def get_output_shape_for(self, input_shape):
        return input_shape[0:-2] + (3,)
