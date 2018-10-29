from keras.engine.saving import load_model
from keras_preprocessing.image import ImageDataGenerator

from src.binned_image_generator import BinnedImageGenerator
from src.model import train_model
from src.util.config import Config
from src.util.util import latest_checkpoint


def train_and_test():
    """
    Train and test in default environment
    """
    # Resume training if checkpoint exists
    checkpoint_dir = latest_checkpoint("colorizer")
    print(f"Latest checkpoint: {checkpoint_dir}")
    model = load_model(str(checkpoint_dir)) if checkpoint_dir is not None else None

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