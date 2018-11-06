from keras_preprocessing.image import DirectoryIterator
from numpy import array, zeros, empty
from skimage.color import rgb2lab

from src.lab_bin_converter import find_bin, bin_to_index


class BinnedImageGenerator(DirectoryIterator):
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

    def _get_batches_of_transformed_samples(self, index_array):
        batch = super(BinnedImageGenerator, self)._get_batches_of_transformed_samples(index_array)

        # Convert batch to lab color space
        batch *= 1.0 / 256.0
        batch_lab = array(list(map(lambda image: rgb2lab(image), batch)))

        # Pull luminance as source
        batch_x = batch_lab[:, :, :, 0:1]
        batch_x *= 1.0 / 100.0

        # Discretize other dimensions
        batch_y_bins = find_bin(batch_lab[:, :, :, 1], batch_lab[:, :, :, 2])
        batch_y_categories = bin_to_index[batch_y_bins]
        oob_mask_cats = batch_y_categories == -1
        oob_batch = batch[oob_mask_cats]

        # Resample image and generate softmax style encoding
        distributions = zeros((batch.shape[0], 64, 64, 313))
        for batch_i in range(batch.shape[0]):
            for x in range(64):
                for y in range(64):
                    for x_offset in [0, 1, 2, 3]:
                        for y_offset in [0, 1, 2, 3]:
                            category = batch_y_categories[batch_i, x * 4 + x_offset, y * 4 + y_offset]
                            assert (category != -1)
                            distributions[batch_i, x, y, category] += 1.0 / 16.0

        batch_y = {
            "color_loss_3": distributions,
            "color_regularizer": empty((batch.shape[0], 0)),
            "lab_coherent": batch_lab
        }
        # TODO make sure lab in range -128,128
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
