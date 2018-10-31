import keras.backend as k
import tensorflow as tf
from keras.engine import Layer
from tensorflow import Tensor


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
        color_classes_flat = k.reshape(color_classes, (-1, k.shape(color_classes)[-1]))
        color_indices = k.argmax(color_classes_flat, -1)
        ab_colors = tf.gather(self.color_map, color_indices)
        # TODO replace hardcoded shape
        ab_colors = tf.reshape(ab_colors, (-1, 256, 256, 2))
        lab_colors = tf.concat([grayscale, ab_colors], axis=-1)

        return lab_colors

    def get_output_shape_for(self, input_shape):
        return input_shape[0:-2] + (3,)
