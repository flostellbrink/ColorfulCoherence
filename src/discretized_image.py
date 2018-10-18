from keras.utils import to_categorical
from keras_preprocessing.image import DirectoryIterator
from numpy import array, zeros, any, all, floor
from skimage.color import rgb2lab, lab2rgb

# Create map of in gamut indexes for lab space
grid_space = 10
grid_range = 220
grid_steps = grid_range // grid_space
lab_colors = zeros(((grid_steps + 1) ** 2, 100, 3))
for l in range(100):
    for a in range(grid_steps + 1):
        for b in range(grid_steps + 1):
            lab_colors[a * (grid_steps + 1) + b, l, ...] = \
                [l, a * grid_space - grid_range // 2, b * grid_space - grid_range // 2]
rgb_colors = lab2rgb(lab_colors)


def rgb_valid(a_, b_):
    value = rgb_colors[a_ * (grid_steps + 1) + b_]
    return any(all((value > 0) & (value < 1), axis=1))


index_map = zeros((grid_steps ** 2,), dtype=int)
index = 0
for a in range(grid_steps):
    for b in range(grid_steps):
        if rgb_valid(a, b) or rgb_valid(a + 1, b) or rgb_valid(a, b + 1) or rgb_valid(a + 1, b + 1):
            index_map[a * grid_steps + b] = index
            index += 1
        else:
            index_map[a * grid_steps + b] = -1

# TODO this is not the expected value of 313!
print(index)


class DiscretizedImageIterator(DirectoryIterator):
    """
    Produces bw and discretized images.
    """

    def __init__(self, directory, image_data_generator,
                 target_size=(256, 256),
                 batch_size=32, shuffle=True, seed=None,
                 data_format='channels_last',
                 follow_links=False,
                 subset=None,
                 interpolation='nearest',
                 dtype='float32'):
        super(DiscretizedImageIterator, self).__init__(directory, image_data_generator,
                                                       target_size, "rgb",
                                                       [""], None,
                                                       batch_size, shuffle, seed,
                                                       data_format,
                                                       None, "", "png",
                                                       follow_links,
                                                       subset,
                                                       interpolation,
                                                       dtype)

    def _get_batches_of_transformed_samples(self, index_array):
        batch = super(DiscretizedImageIterator, self)._get_batches_of_transformed_samples(index_array)

        # Convert batch to lab color space
        batch *= 1.0 / 256.0
        batch_lab = array(list(map(lambda image: rgb2lab(image), batch)))

        # Pull luminance as source
        batch_x = batch_lab[:, :, :, 0:1]
        batch_x *= 1.0 / 100.0

        # Discretize other dimensions
        batch_y_a_indexes = floor((batch_lab[:, :, :, 1] + 128) / grid_space)
        batch_y_b_indexes = floor((batch_lab[:, :, :, 2] + 128) / grid_space)
        batch_y_indexes = (batch_y_a_indexes * grid_steps + batch_y_b_indexes).astype(int)
        # TODO investigate why these occur
        batch_y_indexes[batch_y_indexes >= 484] = 0
        batch_y_categories = index_map[batch_y_indexes]

        # Resample image and generate softmax style encoding
        # TODO this is probably slower than it needs to be
        batch_y = zeros((batch.shape[0], 64, 64, 313))
        for batch_i in range(batch.shape[0]):
            for x in range(64):
                for y in range(64):
                    category = batch_y_categories[batch_i, x * 4, y * 4]
                    if category == -1:
                        # TODO handle slight misses
                        # TODO sample from other nearby pixels
                        # TODO normalize result
                        continue
                    batch_y[batch_i, x, y, category] = 1

        return batch_x, batch_y

    def next(self):
        """For python 2.x.

        # Returns
            The next batch.
        """
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)
