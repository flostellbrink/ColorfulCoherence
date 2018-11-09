import tensorflow as tf

from src.binned_image_generator import BinnedImageGenerator
from src.util.config import Config
from src.util.util import softmax


class ColorfulLoss:
    def __init__(self, generator: BinnedImageGenerator):
        """
        Find empirical color distribution in dataset
        """
        print("Finding empirical color distribution")
        self.distribution = generator.get_bin_counts()
        self.distribution /= max(self.distribution)
        self.distribution = tf.convert_to_tensor(self.distribution)

    def get_loss(self, mix=0.5):
        """
        Returns cross entropy between two distributions, balances classes using the saved distribution.
        :param mix: How much of a uniform distribution to use when mixing it with the empirical distribution
        :return: Loss function
        """
        def loss(yTrue, yPred):
            # Flatten both inputs
            yTrue = tf.reshape(yTrue, (-1, tf.shape(yTrue)[-1]))
            yPred = tf.reshape(yPred, (-1, tf.shape(yPred)[-1]))

            # Compute softmax of prediction
            yPred = softmax(yPred)

            # Find cross entropy across last dimension
            cross_entropy = -tf.reduce_sum((yTrue * tf.log(tf.abs(yPred) + Config.epsilon)), axis=-1)

            # Find weighing term by combining empirical and normal distribution using mix
            weight = (1-mix) * tf.reduce_sum(yTrue * self.distribution, axis=-1) + mix / 313

            return tf.reduce_mean(weight * cross_entropy)

        return loss
