import unittest
from os import environ

from tensorflow import convert_to_tensor, Session

from src.color_regularizer import ColorRegularizer

environ['CUDA_VISIBLE_DEVICES'] = '-1'


class TestColorRegularizer(unittest.TestCase):
    def test_color_regularizer_1(self):
        regularizer = ColorRegularizer()

        # (batch_size, width, height, colors_distribution)
        original = convert_to_tensor([[[[0.0, 3.0, 1.0]]]])
        modified = convert_to_tensor([[[[0.0, 2.0, 1.0]]]])
        with Session() as session:
            regularized = session.run(regularizer.call([original, modified]))
        self.assertAlmostEqual(0.0, regularized)

    def test_color_regularizer_2(self):
        regularizer = ColorRegularizer()

        # (batch_size, width, height, colors_distribution)
        original = convert_to_tensor([[[[1.0, 3.0, 1.0]]]])
        modified = convert_to_tensor([[[[1.0, 1.0, 2.0]]]])
        with Session() as session:
            regularized = session.run(regularizer.call([original, modified]))
        self.assertAlmostEqual(2.0/3.0, regularized)

    def test_color_regularizer_3(self):
        regularizer = ColorRegularizer()

        # (batch_size, width, height, colors_distribution)
        original = convert_to_tensor([[[[1.0, 10.0, 1.0]]]])
        modified = convert_to_tensor([[[[0.0, 0.0, 2.0]]]])
        with Session() as session:
            regularized = session.run(regularizer.call([original, modified]))
        self.assertAlmostEqual(99.0/100.0, regularized)

    def test_color_regularizer_4(self):
        regularizer = ColorRegularizer()

        # (batch_size, width, height, colors_distribution)
        original = convert_to_tensor([[[[1.0, 3.0, 1.0], [1.0, 3.0, 1.0]], [[1.0, 3.0, 1.0], [1.0, 2.0, 1.0]]]])
        modified = convert_to_tensor([[[[1.0, 3.0, 1.0], [1.0, 2.0, 1.0]], [[2.0, 1.0, 1.0], [2.0, 2.0, 3.0]]]])
        with Session() as session:
            regularized = session.run(regularizer.call([original, modified]))
        self.assertAlmostEqual(2.0/3.0 + 1.0/2.0, regularized)


if __name__ == '__main__':
    unittest.main()
