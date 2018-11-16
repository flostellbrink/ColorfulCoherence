from typing import Tuple

from numpy import floor, zeros, ndarray, full, argmax, where

from src.util.util import full_rgb2lab

grid_space = 10
lab_channel_size = 256
grid_steps = lab_channel_size // grid_space


def find_bin(a: ndarray, b: ndarray) -> ndarray:
    """
    Find bins for a and b components of lab
    :param a: A component from -128 to 128
    :param b: B component from -128 to 128
    :return: Bin from 0 to 313
    """
    return (floor((a + (lab_channel_size // 2)) / grid_space) * grid_steps
            + floor((b + (lab_channel_size // 2)) / grid_space)).astype(int)


def build_bins():
    # Create array of all rgb colors
    rgb_channel_size = 256
    rgb_values = zeros((rgb_channel_size, rgb_channel_size, rgb_channel_size, 3))
    for r in range(rgb_channel_size):
        for g in range(rgb_channel_size):
            for b in range(rgb_channel_size):
                rgb_values[r, g, b] = [r, g, b]

    # Convert rgb to lab
    lab_values = full_rgb2lab(rgb_values.reshape(1, -1, 3)).reshape(-1, 3)

    # Find bins
    lab_bin_indexes = find_bin(lab_values[:, 1], lab_values[:, 2])
    lab_bins = full((int(lab_bin_indexes.max()) + 1), -1, dtype=int)
    for index in lab_bin_indexes:
        lab_bins[index] = 1

    # Assign indexes to bins
    index = 0
    for i, is_bin in enumerate(lab_bins):
        if is_bin == 1:
            lab_bins[i] = index
            index = index + 1

    # TODO according to the paper this should be 313, not 262
    print(f"Found {index} color classes in gamut.")
    return lab_bins


bin_to_index_map = build_bins()


def bin_to_ab(bin: int) -> Tuple[int, int]:
    return (bin // grid_steps) * grid_space - (lab_channel_size // 2),\
           (bin % grid_steps) * grid_space - (lab_channel_size // 2)


def build_reverse_bins():
    result = zeros((313, 2))
    for index in range(bin_to_index_map.max()):
        bin = where(bin_to_index_map == index)[0][0]
        a, b = bin_to_ab(bin)
        result[index] = [a, b]

    return result


index_to_lab_map = build_reverse_bins()


def index_to_lab(indices, original, size=256):
    lab_result = zeros((size, size, 3))
    for x in range(size):
        for y in range(size):
            [a, b] = index_to_lab_map[indices[x, y]]
            luminance = original[x, y, 0]
            lab_result[x, y] = [luminance, a, b]
    return lab_result


def mode_to_lab(distribution, original, size=256):
    indices = argmax(distribution, axis=-1)
    return index_to_lab(indices, original, size)


def mean_to_lab(distribution, original):
    lab_result = zeros((256, 256, 3))
    for x in range(256):
        for y in range(256):
            lab_result[x, y] = [original[x, y, 0], 0.0, 0.0]
            for index in range(313):
                [a, b] = index_to_lab_map[index] * distribution[x, y, index]
                lab_result[x, y] += [0.0, a, b]
    return lab_result
