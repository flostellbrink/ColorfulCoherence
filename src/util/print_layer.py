import tensorflow as tf
from keras.engine import Layer


class PrintLayer(Layer):
    """
    Helper layer that prints its input and its shape.
    """

    def __init__(self, **kwargs):
        super(PrintLayer, self).__init__(**kwargs)

    def call(self, x, mask=None):
        return tf.Print(x, [tf.shape(x), x])

    def get_output_shape_for(self, input_shape):
        return input_shape
