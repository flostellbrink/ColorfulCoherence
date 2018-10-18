from keras import Input, Model
from keras.activations import relu, softmax
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, BatchNormalization, Conv2DTranspose, Reshape, Activation
from keras_preprocessing.image import ImageDataGenerator

from src.discretized_image import DiscretizedImageIterator
from src.util.config import Config
from src.util.util import tensor_board


def create_model():
    bw_input = Input(shape=(256, 256, 1))

    conv1_1 = Conv2D(64, 3, name="conv1_1", padding="same", activation=relu)(bw_input)
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

    # For cross entropy: Convert last layer to 2D and activate with softmax
    loss_1 = Reshape((64 * 64, 313))(conv8_313)
    loss_2 = Activation(softmax)(loss_1)

    model = Model(bw_input, loss_2)

    model.compile(optimizer="Adam", loss="binary_crossentropy")
    print(model.summary())
    return model


def train_model(train_generator, test_generator):
    model = create_model()

    model.fit_generator(
        train_generator,
        epochs=100,
        validation_data=test_generator,
        callbacks=[
            tensor_board("auto classifier"),
            EarlyStopping(min_delta=0.001, patience=32)
        ]
    )


def train_and_test():
    data_generator = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        validation_split=0.3)

    train_generator = DiscretizedImageIterator(
        str(Config.data_folder),
        data_generator,
        target_size=(256, 256),
        batch_size=32,
        subset="training")

    test_generator = DiscretizedImageIterator(
        str(Config.data_folder),
        data_generator,
        target_size=(256, 256),
        batch_size=32,
        subset="validation")

    train_model(train_generator, test_generator)


if __name__ == "__main__":
    train_and_test()