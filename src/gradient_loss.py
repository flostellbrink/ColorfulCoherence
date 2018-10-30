import keras.backend as k
import tensorflow as tf
from tensorflow.python.ops.image_ops_impl import image_gradients


def get_gradients(y):
    """
    Get size (manhattan distance) of gradients of the ab channels in all directions
    """
    # (batch_size, width, height, lab)
    y = y[:, :, :, 1:]
    delta_x, delta_y = image_gradients(y)
    return tf.abs(delta_x) + tf.abs(delta_y)


def gradient_loss(yTrue, yPred):
    """
    Sums up the difference in the size of gradients of the ab channels between both inputs
    """
    return k.sum(k.abs(get_gradients(yTrue) - get_gradients(yPred)))