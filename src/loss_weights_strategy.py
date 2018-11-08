from keras.backend import variable, set_value
from keras.callbacks import Callback


class LossWeightsStrategy(Callback):
    """
    Implements training strategy.
    First: Train regularizer, it starts with very low values and can easily be overpowered.
    Second: Add color and coherence training, turn regularizer loss down.
    """
    def __init__(self):
        super().__init__()
        self.color_weight = variable(0.0)
        self.regularizer_weight = variable(0.0)
        self.coherence_weight = variable(0.0)
        self.train_all = False

    def get_weights(self):
        return [self.color_weight, self.regularizer_weight, self.coherence_weight]

    def on_batch_end(self, batch, logs=None):
        if self.train_all:
            return

        # Look at regularizer loss, scale its weight to get decent gradients.
        regularizer_loss = logs['color_regularizer_loss']
        set_value(self.regularizer_weight, 1.0 / abs(regularizer_loss))

        # If the regularizer is large enough lock its weight and enable rest of training.
        if regularizer_loss > 0.1:
            print("Enabling color and coherence training")
            set_value(self.color_weight, 1.0)
            set_value(self.regularizer_weight, 10.0)
            set_value(self.coherence_weight, 1.0)
            self.train_all = True
