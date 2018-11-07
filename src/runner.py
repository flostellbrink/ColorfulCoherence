# Load config to ensure gpu usage is configured
import src.util.config

from keras.engine.saving import load_model
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.python.debug import TensorBoardDebugWrapperSession, LocalCLIDebugWrapperSession

from src.binned_image_generator import BinnedImageGenerator
from src.model import train_model
from src.util.config import Config
from src.util.util import latest_checkpoint

import tensorflow as tf
from keras import backend as k


def train_and_test(resume_training=False, tensorboard_debug=False, cli_debug=False):
    """
    Train and test in default environment
    """
    if tensorboard_debug:
        # Open tf debug session connected to tensor board, this only really works well on linux
        k.set_session(TensorBoardDebugWrapperSession(tf.Session(), '127.0.0.1:6064'))
    elif cli_debug:
        # Open tf debug session with local cli, run manually via ssh
        k.set_session(LocalCLIDebugWrapperSession(tf.Session()))

    if resume_training:
        checkpoint_dir = latest_checkpoint("colorizer")
        print(f"Latest checkpoint: {checkpoint_dir}")
        model = load_model(str(checkpoint_dir)) if checkpoint_dir is not None else None
    else:
        model = None

    # Initialize image generators
    data_generator = ImageDataGenerator(validation_split=0.3)

    train_generator = BinnedImageGenerator(
        str(Config.data_folder),
        data_generator,
        target_size=(256, 256),
        batch_size=Config.batch_size,
        shuffle=True,
        subset="training")

    test_generator = BinnedImageGenerator(
        str(Config.data_folder),
        data_generator,
        target_size=(256, 256),
        batch_size=Config.batch_size,
        subset="validation")

    # Start training
    train_model(train_generator, test_generator, model)


if __name__ == "__main__":
    train_and_test()