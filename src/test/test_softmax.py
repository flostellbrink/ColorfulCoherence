import unittest
from os import environ

from tensorflow import convert_to_tensor, Session

from src.util.util import softmax, softmax_temperature

environ['CUDA_VISIBLE_DEVICES'] = '-1'


class TestSoftmax(unittest.TestCase):
    def test_softmax(self):
        with Session() as session:
            original = convert_to_tensor([1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0])
            result = session.run(softmax(original))
        print([0.02364054, 0.06426166, 0.1746813, 0.474833, 0.02364054, 0.06426166, 0.1746813])
        print(result)

    def test_softmax_temperature(self):
        # This is actually not all that similar to softmax, thanks to stability hacks
        with Session() as session:
            original = convert_to_tensor([1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0])
            result = session.run(softmax_temperature(original, 1.0))
        print([0.02364054, 0.06426166, 0.1746813, 0.474833, 0.02364054, 0.06426166, 0.1746813])
        print(result)

    def test_one_hot_softmax(self):
        with Session() as session:
            original = convert_to_tensor([1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0])
            result = session.run(softmax_temperature(original))
        print([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        print(result)


if __name__ == '__main__':
    unittest.main()
