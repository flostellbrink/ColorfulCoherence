import unittest
from os import environ

from tensorflow import convert_to_tensor, Session

from src.color_regularizer import ColorRegularizer
from src.gradient_loss import gradient_loss

environ['CUDA_VISIBLE_DEVICES'] = '-1'


class TestGradientLoss(unittest.TestCase):
    def test_gradient_loss_1(self):
        """
        Tests that no change is perceived the same (assert that luminance is ignored)
        :return:
        """
        # (batch_size, width, height, lab)
        original = convert_to_tensor([[[[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]], [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]]]])
        modified = convert_to_tensor([[[[0.0, 1.0, 2.0], [2.0, 1.0, 2.0]], [[4.0, 1.0, 2.0], [8.0, 1.0, 2.0]]]])
        with Session() as session:
            regularized = session.run(gradient_loss(original, modified))
        self.assertAlmostEqual(0.0, regularized)

    def test_gradient_loss_2(self):
        # (batch_size, width, height, lab)
        original = convert_to_tensor([[[[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]], [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]]]])
        modified = convert_to_tensor([[[[0.0, 2.0, 2.0], [0.0, 1.0, 2.0]], [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]]]])
        with Session() as session:
            regularized = session.run(gradient_loss(original, modified))
        self.assertAlmostEqual(2.0, regularized)

    def test_gradient_loss_3(self):
        # (batch_size, width, height, lab)
        original = convert_to_tensor([[[[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]], [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]]]])
        modified = convert_to_tensor([[[[0.0, 1.0, 1.0], [0.0, 1.0, 2.0]], [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]]]])
        with Session() as session:
            regularized = session.run(gradient_loss(original, modified))
        self.assertAlmostEqual(2.0, regularized)

    def test_gradient_loss_4(self):
        # (batch_size, width, height, lab)
        original = convert_to_tensor([[[[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]], [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]]]])
        modified = convert_to_tensor([[[[0.0, 2.0, 1.0], [0.0, 1.0, 2.0]], [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]]]])
        with Session() as session:
            regularized = session.run(gradient_loss(original, modified))
        self.assertAlmostEqual(4.0, regularized)

    def test_gradient_loss_5(self):
        # (batch_size, width, height, lab)
        original = convert_to_tensor([[[[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]], [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]]]])
        modified = convert_to_tensor([[[[0.0, 2.0, 1.0], [0.0, 2.0, 1.0]], [[0.0, 2.0, 1.0], [0.0, 2.0, 1.0]]]])
        with Session() as session:
            regularized = session.run(gradient_loss(original, modified))
        self.assertAlmostEqual(0.0, regularized)


if __name__ == '__main__':
    unittest.main()
