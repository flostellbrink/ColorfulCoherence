from argparse import ArgumentParser
from os import environ

import matplotlib.pyplot as plt
from keras import Model
from keras_preprocessing.image import ImageDataGenerator
from numpy import squeeze
from skimage.color import lab2rgb

from src.binned_image_generator import BinnedImageGenerator
from src.colorful_loss import ColorfulLoss
from src.model import create_model
from src.util.config import Config

# Do not use GPU for testing
print("Disabling GPU")
environ['CUDA_VISIBLE_DEVICES'] = '-1'


def show_predictions(generator, model):
    batch_x, batch_y = next(generator)
    prediction = model.predict(batch_x, batch_size=Config.batch_size)

    for batch_id in range(Config.batch_size):
        fig, axes = plt.subplots(ncols=4, figsize=(8, 2))

        axes[0].set_title("Original")
        axes[0].imshow(lab2rgb(batch_y['lab_coherent'][batch_id]))

        axes[1].set_title("Grayscale")
        axes[1].imshow(squeeze(batch_x[batch_id]), cmap='gray')

        axes[2].set_title("Colorful Prediction")
        axes[2].imshow(lab2rgb(prediction[0][batch_id]))

        axes[3].set_title("Coherent Prediction")
        axes[3].imshow(lab2rgb(prediction[1][batch_id]))

        plt.show()


def validate(modelfile):
    """
    Train and test in default environment
    """
    # Initialize image generators
    data_generator = ImageDataGenerator(validation_split=0.3)

    test_generator = BinnedImageGenerator(
        str(Config.data_folder),
        data_generator,
        target_size=(256, 256),
        batch_size=Config.batch_size,
        subset="validation")

    # Create model and load weights from previous run
    colorful_loss = ColorfulLoss(test_generator)
    model, lab_colorful, lab_coherent = create_model(colorful_loss)
    model.load_weights(modelfile)

    # Create new model to output the lab-color layers
    model = Model(inputs=model.input, outputs=[lab_colorful, lab_coherent])

    for _ in range(10):
        show_predictions(test_generator, model)


if __name__ == "__main__":
    parser = ArgumentParser("Test a model")
    parser.add_argument('modelfile', type=str, help="Trained model file")
    args = parser.parse_args()

    print("Reading model from: " + args.modelfile)
    validate(args.modelfile)