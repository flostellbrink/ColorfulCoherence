from argparse import ArgumentParser
from os import environ

import matplotlib.pyplot as plt
from keras import Model
from keras_preprocessing.image import ImageDataGenerator
from numpy import squeeze, zeros, argmax, full
from skimage.color import lab2rgb

from src.binned_image_generator import BinnedImageGenerator
from src.colorful_loss import ColorfulLoss
from src.lab_bin_converter import index_to_lab
from src.model import create_model
from src.util.config import Config

# Do not use GPU for testing
print("Disabling GPU")
environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Set config to validation mode
Config.validation = True


def mode_to_lab(distribution, original):
    lab_result = zeros((256, 256, 3))
    indices = argmax(distribution, axis=-1)
    for x in range(256):
        for y in range(256):
            [a, b] = index_to_lab[indices[x, y]]
            luminance = original[x, y, 0]
            lab_result[x, y] = [luminance, a, b]
    return lab_result


def mean_to_lab(distribution, original):
    lab_result = zeros((256, 256, 3))
    for x in range(256):
        for y in range(256):
            lab_result[x, y] = [original[x, y, 0], 0.0, 0.0]
            for index in range(313):
                [a, b] = index_to_lab[index] * distribution[x, y, index]
                lab_result[x, y] += [0.0, a, b]
    return lab_result


def show_predictions(generator, model):
    batch_x, batch_y = next(generator)
    prediction = model.predict(batch_x, batch_size=Config.batch_size)

    for batch_id in range(Config.batch_size):
        fig, axes = plt.subplots(nrows=2, ncols=6, figsize=(24, 8))

        original_lab = batch_y['lab_coherent'][batch_id]
        axes[0, 0].set_title("Original")
        axes[0, 0].imshow(lab2rgb(original_lab))

        original_luminance = zeros((256, 256, 3))
        original_luminance[:, :, 0] = squeeze(batch_x[batch_id])
        axes[0, 1].set_title("Grayscale")
        axes[0, 1].imshow(lab2rgb(original_luminance), cmap='gray')

        lab_colorful = prediction[0][batch_id]
        axes[0, 2].set_title("Colorful")
        axes[0, 2].imshow(lab2rgb(lab_colorful))

        dist_colorful = prediction[1][batch_id]
        dist_colorful = mode_to_lab(dist_colorful, original_lab)
        axes[0, 3].set_title("Colorful Mode")
        axes[0, 3].imshow(lab2rgb(dist_colorful))

        lab_coherent = prediction[2][batch_id]
        axes[0, 4].set_title("Coherent")
        axes[0, 4].imshow(lab2rgb(lab_coherent))

        dist_coherent = prediction[3][batch_id]
        dist_coherent = mode_to_lab(dist_coherent, original_lab)
        axes[0, 5].set_title("Coherent Mode")
        axes[0, 5].imshow(lab2rgb(dist_coherent))

        original_lab[:, :, 0] = 50
        axes[1, 0].set_title("Original L=50")
        axes[1, 0].imshow(lab2rgb(original_lab))

        original_luminance[:, :, 0] = 50
        axes[1, 1].set_title("Grayscale L=50")
        axes[1, 1].imshow(lab2rgb(original_luminance))

        lab_colorful[:, :, 0] = 50
        axes[1, 2].set_title("Colorful L=50")
        axes[1, 2].imshow(lab2rgb(lab_colorful))

        dist_colorful[:, :, 0] = 50
        axes[1, 3].set_title("Colorful Mode L=50")
        axes[1, 3].imshow(lab2rgb(dist_colorful))

        lab_coherent[:, :, 0] = 50
        axes[1, 4].set_title("Coherent L=50")
        axes[1, 4].imshow(lab2rgb(lab_coherent))

        dist_coherent[:, :, 0] = 50
        axes[1, 5].set_title("Coherent Mode L=50")
        axes[1, 5].imshow(lab2rgb(dist_coherent))

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
    model, lab_colorful, dist_colorful, lab_coherent, dist_coherent = create_model(colorful_loss)
    model.load_weights(modelfile)

    # Create new model to output the lab-color layers
    model = Model(inputs=model.input, outputs=[lab_colorful, dist_colorful, lab_coherent, dist_coherent])

    for _ in range(10):
        show_predictions(test_generator, model)


if __name__ == "__main__":
    parser = ArgumentParser("Test a model")
    parser.add_argument('modelfile', type=str, help="Trained model file")
    args = parser.parse_args()

    print("Reading model from: " + args.modelfile)
    validate(args.modelfile)