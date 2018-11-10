from typing import Tuple

from numpy import floor, zeros, ndarray, full

from src.util.util import full_rgb2lab

grid_space = 10
grid_steps = 256 // grid_space


def find_bin(a: ndarray, b: ndarray) -> ndarray:
    """
    Find bins for a and b components of lab
    :param a: A component from -128 to 128
    :param b: B component from -128 to 128
    :return: Bin from 0 to 313
    """
    return (floor((a + 128) / grid_space) * grid_steps
            + floor((b + 128) / grid_space)).astype(int)


def build_bins():
    # Create array of all rgb colors
    rgb_values = zeros((256, 256, 256, 3))
    for r in range(256):
        for g in range(256):
            for b in range(256):
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


def bin_to_ab(bin: int) -> Tuple[int, int]:
    return (bin // grid_steps) * grid_space - 128, (bin % grid_steps) * grid_space - 128


def build_reverse_bins():
    result = zeros((313, 2))
    for i in range(313):
        a, b = bin_to_ab(i)
        result[i] = [a, b]

    return result


bin_to_index = build_bins()
index_to_lab = build_reverse_bins()