from pathlib import Path
from typing import Union

import tensorflow as tf
from keras import Model
from keras.backend import zeros_like
from keras.callbacks import TensorBoard, ModelCheckpoint
from skimage.color import rgb2lab, lab2rgb

from src.util.config import Config


class Util:
    def __init__(self, network: str):
        self.network = network
        folder = Config.log_folder.joinpath(network)
        self.latest_folder = sorted(list(map(lambda f: int(f.name), folder.glob("*"))) + [-1])[-1]

    def tensor_board(self) -> TensorBoard:
        """
        Create a tensor board instance in a new sub folder
        """
        folder = Config.log_folder.joinpath(self.network).joinpath(str(self.latest_folder + 1))
        folder.mkdir(parents=True, exist_ok=True)
        print(f"Tensor board folder: {str(folder)}")
        return TensorBoard(log_dir=str(folder))

    def model_checkpoint(self) -> ModelCheckpoint:
        """
        Create checkpoints in new sub folder
        :return:
        """
        folder = Config.model_folder.joinpath(self.network).joinpath(str(self.latest_folder + 1))
        folder.mkdir(parents=True, exist_ok=True)
        file = folder.joinpath("checkpoint-{epoch:02d}-l:{loss:.2f}-vl:{val_loss:.2f}.h5")
        print(f"Checkpoint file: {str(file)}")
        return ModelCheckpoint(str(file))

    def save_model(self, model: Model):
        folder = Config.model_folder.joinpath(self.network).joinpath(str(self.latest_folder + 1))
        folder.mkdir(parents=True, exist_ok=True)
        file = folder.joinpath("model.h5")
        print(f"Model file: {str(file)}")
        model.save(str(file))


def latest_checkpoint(network: str) -> Union[Path, None]:
    folder = Config.model_folder.joinpath(network)
    latest_folder = (sorted(folder.glob("*"), key=lambda f: int(f.name)) + [None])[-1]
    if latest_folder is None:
        return None
    return (sorted(latest_folder.glob("*"), key=lambda file: file.stat().st_mtime) + [None])[-1]


def zero_loss(y_true, y_pred):
    return zeros_like(y_pred)


def identity_loss(y_true, y_pred):
    """
    The prediction is a loss function, returns it.
    """
    return y_pred


def not_zero(x):
    """
    Returns relu, whith small epsilon instead of zero.
    Used to avoid nans in divisions and logs.
    """
    return tf.maximum(x, Config.epsilon)


def softmax(logits):
    """
    Softmax function in last dimension of input.
    """
    exp_logits = tf.exp(tf.sigmoid(logits))
    return exp_logits / not_zero(tf.reshape(tf.reduce_sum(exp_logits, axis=-1), (-1, 1)))


def softmax_temperature(logits, temperature = 0.01):
    """
    Runs softmax with temperature on logits, adds additional hacks to avoid undefinedness of log around 0.
    :param logits: Input distribution
    :param temperature: Temperature. 1 means softmax, towards 0 behaves like one hot
    :return: Softmaxed distribution
    """
    exp_log_by_temp = tf.exp(tf.log(tf.sigmoid(logits) + Config.epsilon) / temperature)
    return exp_log_by_temp / not_zero(tf.reshape(tf.reduce_sum(exp_log_by_temp, axis=-1), (-1, 1)))


def full_rgb2lab(rgb):
    """
    The inner function converts from [0,1] to proper lab.
    This convert from [0,255] to proper lab.
    """
    return rgb2lab(rgb / 255.0)
