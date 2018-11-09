import tensorflow as tf
from keras.engine import Layer

from src.util.util import softmax_temperature, softmax, not_zero


class ColorRegularizer(Layer):
    """
    Regularizer makes sure that the argmax of a modified distribution has a high value in the original distribution.
    This isn't implemented as a regularizer because it needs input from two layers. It's cleaner to do this in a layer.
    To avoid optimizations removing a dangling layer, the output is used as loss.
    """

    def __init__(self, **kwargs):
        super(ColorRegularizer, self).__init__(**kwargs)

    def call(self, x, mask=None):
        [original, boosted] = x

        # Flatten and normalize original distribution
        original = tf.reshape(original, (-1, tf.shape(original)[-1]))
        original = softmax(original)

        # Flatten and approximately one-hot encode modified distribution
        boosted = tf.reshape(boosted, (-1, tf.shape(boosted)[-1]))
        boosted = softmax_temperature(boosted)

        # Multiply new encoding with old element wise.
        # This will only have a high output if the one hot selection is likely in the original.
        multiplied = tf.multiply(original, boosted)

        loss = tf.reduce_sum(multiplied, axis=-1)
        # Correct loss for the assumption that the original is one hot encoded.
        # TODO This seems to encourage small original values and ultimately 0s
        # TODO That would mean, this loss bleads back into the colorful part, which should definetely be stopped by stop gradients!
        # loss /= tf.reduce_max(original, axis=-1)

        # When we have a complete match each sum is 1, we invert this to get a minimizable loss.
        loss = tf.reduce_mean(1.0 - loss)

        return loss

    def get_output_shape_for(self, input_shape):
        return (1)
