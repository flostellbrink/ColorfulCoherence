from pathlib import Path
from typing import Union

import tensorflow as tf
from keras import Model
from keras.backend import zeros_like
from keras.callbacks import TensorBoard, ModelCheckpoint
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
        return TensorBoard(log_dir=str(folder))

    def model_checkpoint(self) -> ModelCheckpoint:
        """
        Create checkpoints in new sub folder
        :return:
        """
        folder = Config.model_folder.joinpath(self.network).joinpath(str(self.latest_folder + 1))
        folder.mkdir(parents=True, exist_ok=True)
        file = folder.joinpath("checkpoint-{epoch:02d}-l:{loss:.2f}-vl:{val_loss:.2f}.h5")
        return ModelCheckpoint(str(file), period=1  )

    def save_model(self, model: Model):
        folder = Config.model_folder.joinpath(self.network).joinpath(str(self.latest_folder + 1))
        folder.mkdir(parents=True, exist_ok=True)
        file = folder.joinpath("model.h5")
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


def softmax_temperature(logits, temperature):
    """
    Runs softmax with temperature on logits
    :param logits: Input distribution
    :param temperature: Temperature. 1 means softmax, towards 0 behaves like one hot
    :return: Softmaxed distribution
    """
    log_by_temp = tf.log(tf.sigmoid(logits) + Config.epsilon) / temperature
    return tf.exp(log_by_temp) / tf.reduce_sum(tf.exp(log_by_temp))