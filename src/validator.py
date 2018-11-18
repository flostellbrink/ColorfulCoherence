from argparse import ArgumentParser
from os import environ

# Avoid no display exception of matplotlib when no display is set
# https://stackoverflow.com/questions/37604289/tkinter-tclerror-no-display-name-and-no-display-environment-variable/43592515
import matplotlib
if environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
from keras import Model
from keras_preprocessing.image import ImageDataGenerator
from numpy import squeeze, zeros, ones, kron

from src.binned_image_generator import BinnedImageGenerator
from src.colorful_loss import ColorfulLoss
from src.lab_bin_converter import mode_to_lab
from src.model import create_model
from src.util.config import Config

from src.util.util import full_lab2rgb

# Do not use GPU for testing
print("Disabling GPU")
Config.enable_gpu = False
environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Set config to validation mode
Config.validation = True


def show_predictions(generator, model, index, render_mode=False):
    batch_x, batch_y = next(generator)
    prediction = model.predict(batch_x, batch_size=Config.batch_size)

    for batch_id in range(Config.batch_size):
        fig, axes = plt.subplots(nrows=2, ncols=5, figsize=((5*4), (2*4)))

        original_lab = batch_y['lab_coherent'][batch_id]
        original_luminance = zeros((256, 256, 3))
        original_luminance[:, :, 0] = squeeze(batch_x[batch_id])
        original_dist = batch_y['dist_colorful'][batch_id]
        original_dist = kron(original_dist, ones((4, 4, 1)))
        original_dist = mode_to_lab(original_dist, original_lab)
        lab_colorful = prediction[0][batch_id]
        dist_colorful = prediction[1][batch_id]
        dist_colorful = mode_to_lab(dist_colorful, original_lab)
        lab_coherent = prediction[2][batch_id]
        dist_coherent = prediction[3][batch_id]
        dist_coherent = mode_to_lab(dist_coherent, original_lab)

        axes[0, 0].set_title("Original")
        axes[0, 0].imshow(full_lab2rgb(original_lab))
        axes[0, 1].set_title("Grayscale")
        axes[0, 1].imshow(full_lab2rgb(original_luminance), cmap='gray')
        axes[0, 2].set_title("Original Discretized")
        axes[0, 2].imshow(full_lab2rgb(original_dist))
        if not render_mode:
            axes[0, 3].set_title("Colorful")
            axes[0, 3].imshow(full_lab2rgb(lab_colorful))
            axes[0, 4].set_title("Coherent")
            axes[0, 4].imshow(full_lab2rgb(lab_coherent))
        else:
            axes[0, 3].set_title("Colorful Mode")
            axes[0, 3].imshow(full_lab2rgb(dist_colorful))
            axes[0, 4].set_title("Coherent Mode")
            axes[0, 4].imshow(full_lab2rgb(dist_coherent))

        original_lab[:, :, 0] = 50
        original_luminance[:, :, 0] = 50
        original_dist[:, :, 0] = 50
        lab_colorful[:, :, 0] = 50
        lab_coherent[:, :, 0] = 50
        dist_colorful[:, :, 0] = 50
        dist_coherent[:, :, 0] = 50

        axes[1, 0].set_title("Original L=50")
        axes[1, 0].imshow(full_lab2rgb(original_lab))
        axes[1, 1].set_title("Grayscale L=50")
        axes[1, 1].imshow(full_lab2rgb(original_luminance))
        axes[1, 2].set_title("Original Discretized L=50")
        axes[1, 2].imshow(full_lab2rgb(original_dist))

        if not render_mode:
            axes[1, 3].set_title("Colorful L=50")
            axes[1, 3].imshow(full_lab2rgb(lab_colorful))
            axes[1, 4].set_title("Coherent L=50")
            axes[1, 4].imshow(full_lab2rgb(lab_coherent))
        else:
            axes[1, 3].set_title("Colorful Mode L=50")
            axes[1, 3].imshow(full_lab2rgb(dist_colorful))
            axes[1, 4].set_title("Coherent Mode L=50")
            axes[1, 4].imshow(full_lab2rgb(dist_coherent))

        Config.output_folder.mkdir(parents=True, exist_ok=True)
        path = Config.output_folder.joinpath(f"example{index}-{batch_id}.svg")
        plt.savefig(str(path), bbox_inches='tight', format="svg")
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

    for index in range(10):
        show_predictions(test_generator, model, index)


if __name__ == "__main__":
    parser = ArgumentParser("Test a model")
    parser.add_argument('modelfile', type=str, help="Trained model file")
    args = parser.parse_args()

    print("Reading model from: " + args.modelfile)
    validate(args.modelfile)
