from keras import Input, Model
from keras.activations import relu, softmax
from keras.backend import stop_gradient
from keras.engine import Layer
from keras.layers import Conv2D, BatchNormalization, Conv2DTranspose, Reshape, Activation, Lambda, UpSampling2D, \
    Concatenate
import tensorflow as tf
from tensorflow import convert_to_tensor, float32

from src.binned_image_generator import BinnedImageGenerator
from src.color_regularizer import ColorRegularizer
from src.dist_to_lab import DistToLab
from src.gradient_loss import gradient_loss
from src.lab_bin_converter import index_to_lab
from src.util.config import Config
from src.util.print_layer import PrintLayer
from src.util.util import Util, zero_loss, identity_loss


def create_color_model(grayscale_input: Layer) -> Layer:
    conv1_1 = Conv2D(64, 3, name="conv1_1", padding="same", activation=relu)(grayscale_input)
    conv1_2 = Conv2D(64, 3, name="conv1_2", padding="same", strides=2, activation=relu)(conv1_1)
    conv1_2norm = BatchNormalization(name="conv1_2norm")(conv1_2)

    conv2_1 = Conv2D(128, 3, name="conv2_1", padding="same", activation=relu)(conv1_2norm)
    conv2_2 = Conv2D(128, 3, name="conv2_2", padding="same", strides=2, activation=relu)(conv2_1)
    conv2_2norm = BatchNormalization(name="conv2_2norm")(conv2_2)

    conv3_1 = Conv2D(256, 3, name="conv3_1", padding="same", activation=relu)(conv2_2norm)
    conv3_2 = Conv2D(256, 3, name="conv3_2", padding="same", strides=2, activation=relu)(conv3_1)
    conv3_2norm = BatchNormalization(name="conv3_2norm")(conv3_2)

    conv4_1 = Conv2D(512, 3, name="conv4_1", padding="same", activation=relu)(conv3_2norm)
    conv4_2 = Conv2D(512, 3, name="conv4_2", padding="same", activation=relu)(conv4_1)
    conv4_2norm = BatchNormalization(name="conv4_2norm")(conv4_2)

    conv5_1 = Conv2D(512, 3, name="conv5_1", padding="same", dilation_rate=2, activation=relu)(conv4_2norm)
    conv5_2 = Conv2D(512, 3, name="conv5_2", padding="same", dilation_rate=2, activation=relu)(conv5_1)
    conv5_3 = Conv2D(512, 3, name="conv5_3", padding="same", dilation_rate=2, activation=relu)(conv5_2)
    conv5_3norm = BatchNormalization(name="conv5_3norm")(conv5_3)

    conv6_1 = Conv2D(512, 3, name="conv6_1", padding="same", dilation_rate=2, activation=relu)(conv5_3norm)
    conv6_2 = Conv2D(512, 3, name="conv6_2", padding="same", dilation_rate=2, activation=relu)(conv6_1)
    conv6_3 = Conv2D(512, 3, name="conv6_3", padding="same", dilation_rate=2, activation=relu)(conv6_2)
    conv6_3norm = BatchNormalization(name="conv6_3norm")(conv6_3)

    conv7_1 = Conv2D(512, 3, name="conv7_1", padding="same", activation=relu)(conv6_3norm)
    conv7_2 = Conv2D(512, 3, name="conv7_2", padding="same", activation=relu)(conv7_1)
    conv7_3 = Conv2D(512, 3, name="conv7_3", padding="same", activation=relu)(conv7_2)
    conv7_3norm = BatchNormalization(name="conv7_3norm")(conv7_3)

    conv8_1 = Conv2DTranspose(256, 4, name="conv8_1", padding="same", strides=2, activation=relu)(conv7_3norm)
    conv8_2 = Conv2D(256, 3, name="conv8_2", padding="same", activation=relu)(conv8_1)
    conv8_3 = Conv2D(256, 3, name="conv8_3", padding="same", activation=relu)(conv8_2)
    conv8_3norm = BatchNormalization(name="conv8_3norm")(conv8_3)
    conv8_313 = Conv2D(313, 1, name="conv8_313")(conv8_3norm)

    return conv8_313


def create_coherence_model(grayscale_input: Layer, color_output: Layer)-> Layer:
    conv1_1 = Conv2D(32, 3, name="coh_conv1_1", padding="same", activation=relu)(grayscale_input)
    conv1_2 = Conv2D(32, 3, name="coh_conv1_2", padding="same", activation=relu)(conv1_1)
    conv1_3 = Conv2D(32, 3, name="coh_conv1_3", padding="same", activation=relu)(conv1_2)
    conv1_3norm = BatchNormalization(name="coh_conv1_2norm")(conv1_3)

    # Set gradients to zero to prevent backpropagation into color model
    color_output = Lambda(lambda x: stop_gradient(x), name="stop_color_gradient")(color_output)
    concat = Concatenate(name="concat")([conv1_3norm, color_output])
    conv2_1 = Conv2D(313, 3, name="coh_conv2_1", padding="same")(concat)

    return conv2_1


def create_model()-> Model:
    grayscale_input = Input(shape=(256, 256, 1))
    dist_colorful = create_color_model(grayscale_input)
    # TODO interpolation?
    up_sample_colorful = UpSampling2D((4, 4), name="up_sample_colorful")(dist_colorful)
    dist_coherent = create_coherence_model(grayscale_input, up_sample_colorful)

    # For color cross entropy: Convert last color layer to 2D and activate with softmax
    color_loss_1 = Reshape((64 * 64, 313), name="color_loss_1")(dist_colorful)
    color_loss_2 = Activation(softmax, name="color_loss_2")(color_loss_1)
    color_loss_3 = Reshape((64, 64, 313), name="color_loss_3")(color_loss_2)

    # Regularizer to keep colors when optimizing coherence
    color_regularizer = ColorRegularizer(name="color_regularizer")([up_sample_colorful, dist_coherent])

    # Create lab space images (for comparison and gradient loss)
    color_map = convert_to_tensor(index_to_lab, dtype=float32)
    lab_colorful = DistToLab(color_map, name="lab_colorful")([grayscale_input, up_sample_colorful])
    lab_coherent = DistToLab(color_map, name="lab_coherent")([grayscale_input, dist_coherent])

    model = Model(grayscale_input, [color_loss_3, color_regularizer, lab_coherent])
    losses = {
        "color_loss_3": "categorical_crossentropy",
        "color_regularizer": identity_loss,
        "lab_coherent": gradient_loss
    }
    model.compile(optimizer="Adam", loss=losses)
    print(model.summary())
    return model


def train_model(train_generator: BinnedImageGenerator, test_generator: BinnedImageGenerator, model: Model=None):
    if model is None:
        print("Creating fresh model...")
        model = create_model()

    util = Util("colorizer")
    model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=Config.max_epochs,
        validation_data=test_generator,
        validation_steps=len(test_generator),
        callbacks=[
            util.tensor_board(),
            util.model_checkpoint()
        ]
    )

    util.save_model(model)
