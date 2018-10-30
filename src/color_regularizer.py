import keras.backend as k
import tensorflow as tf
from keras.engine import Layer


class ColorRegularizer(Layer):
    """
    Regularizer makes sure that the argmax of a modified distribution has a high value in the original distribution.
    """

    def __init__(self, **kwargs):
        super(ColorRegularizer, self).__init__(**kwargs)

    def call(self, x, mask=None):
        original = x[0]
        original = k.reshape(original, (-1, k.shape(original)[-1]))
        boosted = x[1]
        boosted = k.reshape(boosted, (-1, k.shape(boosted)[-1]))
        boosted_indices = tf.stack([tf.range(boosted.shape[0], dtype=tf.int64), k.argmax(boosted, -1)], axis=-1)
        original_lookup = tf.gather_nd(original, indices=boosted_indices)
        original_max = k.max(original, -1)
        loss = k.sum(1 - original_lookup / original_max)

        self.add_loss(loss, x)
        return loss

    def get_output_shape_for(self, input_shape):
        return (1)
