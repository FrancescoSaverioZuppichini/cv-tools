import unittest

import numpy as np

from cv_tools.sam3 import encode_rle


def decode_rle(rle):
    flat = np.zeros(rle["size"][0] * rle["size"][1], dtype=bool)
    starts = np.cumsum([0] + rle["counts"][:-1])
    for start, count, value in zip(starts, rle["counts"], np.arange(len(rle["counts"])) % 2):
        flat[start:start + count] = value
    return flat.reshape(rle["size"], order="F")


class RleTests(unittest.TestCase):
    def test_round_trip_matches_coco_layout(self):
        rng = np.random.default_rng(0)
        for mask in [rng.random((7, 5)) > .5, np.ones((3, 4), bool), np.zeros((3, 4), bool)]:
            rle = encode_rle(mask)
            np.testing.assert_array_equal(decode_rle(rle), mask)
            self.assertEqual(sum(rle["counts"]), mask.size)
        self.assertEqual(encode_rle(np.ones((2, 2), bool))["counts"], [0, 4])


if __name__ == "__main__":
    unittest.main()
