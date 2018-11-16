import unittest
from random import sample

from numpy import array, full

from src.lab_bin_converter import find_bin, index_to_lab, bin_to_index_map


class TestBins(unittest.TestCase):
    def test_conversion(self):
        input_a = array(list(sample(range(-50, 50, 1), 16))).reshape((4, 4))
        input_b = array(list(sample(range(-50, 50, 1), 16))).reshape((4, 4))

        bins = find_bin(input_a, input_b)
        categories = bin_to_index_map[bins]
        labs = index_to_lab(categories, full((4, 4, 3), 50), size=4)

        input_a = input_a.reshape((16,))
        input_b = input_b.reshape((16,))
        labs = labs.reshape(16, 3)
        for index, lab in enumerate(labs):
            self.assertLessEqual(abs(input_a[index] - lab[1]), 10,
                                 f"in: {input_a[index]}, {input_b[index]}, out: {lab}")
            self.assertLessEqual(abs(input_b[index] - lab[2]), 10,
                                 f"in: {input_a[index]}, {input_b[index]}, out: {lab}")


if __name__ == '__main__':
    unittest.main()
