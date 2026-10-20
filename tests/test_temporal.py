import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from cv_tools.sapiens2.temporal import Frame, Prediction, between, predictions


class FakePose:
    def __init__(self):
        self.batch_sizes = []

    def prepare(self, images):
        self.batch_sizes.append(len(images))
        return images, [[[0, 0, 8, 8]]] * len(images)

    def predict(self, images, boxes):
        return [(np.array([[1 + image[0, 0, 0] * 10, 2.0]], dtype=np.float32), np.array([0.9]))
                for image in images]


def frames_at(times):
    return [Frame(i, t, np.full((8, 8, 3), t, dtype=np.float32)) for i, t in enumerate(times)]


class TemporalTests(unittest.TestCase):
    def test_irregular_timestamps_and_partial_batch(self):
        frames = frames_at([0, .04, .09, .16, .23, .34])
        args = SimpleNamespace(input=Path("clip.mp4"), frames=0, batch_size=2,
                               inference_fps=5, interpolate="linear", threshold=.3)
        pose = FakePose()
        with patch("cv_tools.sapiens2.temporal.read_frames", return_value=iter(frames)):
            output = list(predictions(pose, args, 30))
        self.assertEqual([p.frame.index for p in output], list(range(6)))
        self.assertEqual(pose.batch_sizes, [2, 1])
        self.assertEqual([p.frame.index for p in output if p.kind == "inferred"], [0, 4, 5])
        np.testing.assert_allclose([p.points[0, 0] for p in output], [1 + f.timestamp * 10 for f in frames], atol=1e-6)
        self.assertEqual(output[-1].kind, "inferred")

    def test_sampling_keeps_every_output_frame_and_final_anchor(self):
        for fps in (30, 30000 / 1001):
            for rate, count in [(None, 180), (10, 61), (15, 91), (60, 180)]:
                with self.subTest(source_fps=fps, inference_fps=rate):
                    frames = frames_at(np.arange(180) / fps)
                    args = SimpleNamespace(input=Path("clip.mp4"), frames=0, batch_size=1,
                                           inference_fps=rate, interpolate="linear", threshold=.3)
                    with patch("cv_tools.sapiens2.temporal.read_frames", return_value=iter(frames)):
                        output = list(predictions(FakePose(), args, fps))
                    self.assertEqual([p.frame.index for p in output], list(range(180)))
                    self.assertEqual(sum(p.kind == "inferred" for p in output), count)
                    self.assertEqual(output[-1].kind, "inferred")

    def test_interpolation_rejects_weak_and_out_of_frame_endpoints(self):
        left_frame, middle, right_frame = frames_at([0, .05, .1])
        left = Prediction(left_frame, np.array([[1., 1.], [2., 2.], [3., 3.]]),
                          np.array([.9, .9, .9]), [0, 0, 8, 8], "inferred", (0,))
        right = Prediction(right_frame, np.array([[3., 3.], [4., 4.], [1000., 1000.]]),
                           np.array([.8, .1, .9]), left.box, "inferred", (2,))
        result = between(left, right, middle, .3, "linear")
        np.testing.assert_allclose(result.points[0], [2., 2.])
        self.assertAlmostEqual(result.scores[0], .8)
        self.assertTrue(np.isnan(result.points[1:]).all())
        np.testing.assert_array_equal(result.scores[1:], [0., 0.])
        held = between(left, right, middle, .3, "none")
        self.assertEqual(held.kind, "held")
        np.testing.assert_array_equal(held.points, left.points)

    def test_cpu_output_worker_preserves_order_and_surfaces_errors(self):
        from cv_tools.sapiens2.cli import write_run

        args = SimpleNamespace(input=Path("clip.mp4"), frames=0, batch_size=1,
                               inference_fps=10, interpolate="linear", threshold=.3, device="cpu")
        pose = FakePose()
        pose.selected = [0]
        pose.draw = lambda image, points, scores: (image.astype(np.uint8), image.astype(np.uint8))
        for fail in (False, True):
            with self.subTest(write_failure=fail), tempfile.TemporaryDirectory() as directory:
                writers = [MagicMock(), MagicMock()]
                for writer in writers:
                    writer.isOpened.return_value = True
                if fail:
                    writers[0].write.side_effect = OSError("encoder failed")
                with patch("cv_tools.sapiens2.temporal.read_frames", return_value=iter(frames_at(np.arange(6) / 30))), \
                     patch("cv_tools.sapiens2.cli.cv2.VideoWriter", side_effect=writers):
                    if fail:
                        with self.assertRaisesRegex(OSError, "encoder failed"):
                            write_run(pose, args, Path(directory), 30)
                    else:
                        stats, _ = write_run(pose, args, Path(directory), 30)
                        records = [json.loads(line) for line in Path(directory, "landmarks.jsonl").read_text().splitlines()]
                        self.assertEqual([row["frame"] for row in records], list(range(6)))
                        self.assertEqual(stats["frames"], 6)
                        self.assertEqual(stats["inferred_frames"], 3)
                        self.assertEqual(stats["interpolated_frames"], 3)
                        self.assertTrue(all(writer.write.call_count == 6 for writer in writers))
                for writer in writers:
                    writer.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()
