from keras_preprocessing.image import DirectoryIterator
from numpy import array, zeros, floor, full, ndarray
from skimage.color import rgb2lab


class BinnedImageGenerator(DirectoryIterator):
    """
    Produces bw and discretized images.
    """

    def find_bin(self, a: ndarray, b: ndarray)-> ndarray:
        """
        Find bins for a and b components of lab
        :param a: A component from -128 to 128
        :param b: B component from -128 to 128
        :return: Bin from 0 to 313
        """
        return (floor((a + 128) / self.grid_space) * self.grid_steps
                + floor((b + 128) / self.grid_space)).astype(int)

    def build_bins(self):
        # Create array of all rgb colors
        rgb_values = zeros((256, 256 * 256, 3))
        for r in range(256):
            for g in range(256):
                for b in range(256):
                    rgb_values[r, g * 256 + b] = [r / 256, g / 256, b / 256]

        # Convert rgb to lab
        lab_values = rgb2lab(rgb_values).reshape(-1, 3)

        # Find bins
        lab_bin_indexes = self.find_bin(lab_values[:, 1], lab_values[:, 2])
        lab_bins = full((int(lab_bin_indexes.max()) + 1), -1, dtype=int)
        for index in lab_bin_indexes:
            lab_bins[index] = 1

        # Assign indexes to bins
        index = 0
        for i, is_bin in enumerate(lab_bins):
            if is_bin == 1:
                lab_bins[i] = index
                index = index + 1

        # TODO according to the paper this should be 313, not 260
        print(index)
        return lab_bins

    def __init__(self, directory, image_data_generator,
                 target_size=(256, 256),
                 batch_size=32, shuffle=True, seed=None,
                 data_format='channels_last',
                 follow_links=False,
                 subset=None,
                 interpolation='nearest',
                 dtype='float32'):
        super(BinnedImageGenerator, self).__init__(directory, image_data_generator,
                                                   target_size, "rgb",
                                                   [""], None,
                                                   batch_size, shuffle, seed,
                                                   data_format,
                                                   None, "", "png",
                                                   follow_links,
                                                   subset,
                                                   interpolation,
                                                   dtype)
        self.grid_space = 10
        self.grid_steps = 256 // self.grid_space
        self.lab_bins = self.build_bins()

    def _get_batches_of_transformed_samples(self, index_array):
        batch = super(BinnedImageGenerator, self)._get_batches_of_transformed_samples(index_array)

        # Convert batch to lab color space
        batch *= 1.0 / 256.0
        batch_lab = array(list(map(lambda image: rgb2lab(image), batch)))

        # Pull luminance as source
        batch_x = batch_lab[:, :, :, 0:1]
        batch_x *= 1.0 / 100.0

        # Discretize other dimensions
        batch_y_bins = self.find_bin(batch_lab[:, :, :, 1], batch_lab[:, :, :, 2])
        batch_y_categories = self.lab_bins[batch_y_bins]
        oob_mask_cats = batch_y_categories == -1
        oob_batch = batch[oob_mask_cats]

        # Resample image and generate softmax style encoding
        batch_y = zeros((batch.shape[0], 64, 64, 313))
        for batch_i in range(batch.shape[0]):
            for x in range(64):
                for y in range(64):
                    for x_offset in [0, 1, 2, 3]:
                        for y_offset in [0, 1, 2, 3]:
                            category = batch_y_categories[batch_i, x * 4 + x_offset, y * 4 + y_offset]
                            assert (category != -1)
                            batch_y[batch_i, x, y, category] += 1.0 / 16.0

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
