import keras.backend as k
import tensorflow as tf
from keras.engine import Layer

from src.util.util import softmax_temperature


class ColorRegularizer(Layer):
    """
    Regularizer makes sure that the argmax of a modified distribution has a high value in the original distribution.
    """

    def __init__(self, temperature=0.1, **kwargs):
        self.temperature = temperature
        super(ColorRegularizer, self).__init__(**kwargs)

    def call(self, x, mask=None):
        [original, boosted] = x
        original = k.reshape(original, (-1, k.shape(original)[-1]))
        original = softmax_temperature(original, 1.0)
        boosted = k.reshape(boosted, (-1, k.shape(boosted)[-1]))
        boosted = softmax_temperature(boosted, self.temperature)
        multiplied = tf.multiply(original, boosted)

        # When complete match the sum is 1, to turn this into loss we invert it
        loss = k.sum(1 - tf.reduce_sum(multiplied, axis=-1))

        self.add_loss(loss, x)
        return loss

    def get_output_shape_for(self, input_shape):
        return (1)
