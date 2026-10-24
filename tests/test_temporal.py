import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from cv_tools.media import Frame
from cv_tools.sapiens2.temporal import Prediction, between, predictions


class FakePose:
    def predict(self, image):
        return np.array([[1 + image[0, 0, 0] * 10, 2.0]], dtype=np.float32), np.array([0.9]), [0, 0, 8, 8]


def frames_at(times):
    return [Frame(i, t, np.full((8, 8, 3), t, dtype=np.float32)) for i, t in enumerate(times)]


def run(times, fps, rate):
    args = SimpleNamespace(input=Path("clip.mp4"), is_image=False, frames=0, inference_fps=rate, threshold=.3)
    with patch("cv_tools.sapiens2.temporal.read_frames", return_value=iter(frames_at(times))):
        return list(predictions(FakePose(), args, fps))


class TemporalTests(unittest.TestCase):
    def test_irregular_timestamps_interpolate_linearly(self):
        times = [0, .04, .09, .16, .23, .34]
        output = run(times, 30, 5)
        self.assertEqual([p.frame.index for p in output], list(range(6)))
        self.assertEqual([p.frame.index for p in output if p.kind == "inferred"], [0, 4, 5])
        np.testing.assert_allclose([p.points[0, 0] for p in output], [1 + t * 10 for t in times], atol=1e-6)

    def test_sampling_keeps_every_output_frame_and_final_anchor(self):
        for fps in (30, 30000 / 1001):
            for rate, count in [(None, 180), (10, 61), (15, 91), (60, 180)]:
                with self.subTest(source_fps=fps, inference_fps=rate):
                    output = run(np.arange(180) / fps, fps, rate)
                    self.assertEqual([p.frame.index for p in output], list(range(180)))
                    self.assertEqual(sum(p.kind == "inferred" for p in output), count)
                    self.assertEqual(output[-1].kind, "inferred")

    def test_interpolation_rejects_weak_and_out_of_frame_endpoints(self):
        left_frame, middle, right_frame = frames_at([0, .05, .1])
        left = Prediction(left_frame, np.array([[1., 1.], [2., 2.], [3., 3.]]),
                          np.array([.9, .9, .9]), [0, 0, 8, 8], "inferred", (0,))
        right = Prediction(right_frame, np.array([[3., 3.], [4., 4.], [1000., 1000.]]),
                           np.array([.8, .1, .9]), left.box, "inferred", (2,))
        result = between(left, right, middle, .3)
        np.testing.assert_allclose(result.points[0], [2., 2.])
        self.assertAlmostEqual(result.scores[0], .8)
        self.assertTrue(np.isnan(result.points[1:]).all())
        np.testing.assert_array_equal(result.scores[1:], [0., 0.])


if __name__ == "__main__":
    unittest.main()
