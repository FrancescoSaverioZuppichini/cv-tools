import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from cv_tools.media import write_run


class WriteRunTests(unittest.TestCase):
    def test_output_worker_preserves_order_and_surfaces_errors(self):
        images = [np.full((8, 8, 3), i, dtype=np.uint8) for i in range(6)]

        def render(image, index):
            return {"frame": index, "value": int(image[0, 0, 0])}, {"guide.mp4": image, "overlay.mp4": image}

        for fail in (False, True):
            with self.subTest(write_failure=fail), tempfile.TemporaryDirectory() as directory:
                writers = [MagicMock(), MagicMock()]
                for writer in writers:
                    writer.isOpened.return_value = True
                if fail:
                    writers[0].write.side_effect = OSError("encoder failed")
                with patch("cv_tools.media.cv2.VideoWriter", side_effect=writers):
                    if fail:
                        with self.assertRaisesRegex(OSError, "encoder failed"):
                            write_run(Path(directory), iter(images), render, 30, "frames.jsonl")
                    else:
                        stats = write_run(Path(directory), iter(images), render, 30, "frames.jsonl")
                        records = [json.loads(line) for line in Path(directory, "frames.jsonl").read_text().splitlines()]
                        self.assertEqual([(r["frame"], r["value"]) for r in records], [(i, i) for i in range(6)])
                        self.assertEqual(stats["frames"], 6)
                        self.assertTrue(all(writer.write.call_count == 6 for writer in writers))
                # [Agent] Encoders open lazily on their first frame, so a failure on the first write never opens the second.
                for writer in writers[:1] if fail else writers:
                    writer.release.assert_called_once()

    def test_still_outputs_by_extension(self):
        with tempfile.TemporaryDirectory() as directory:
            depth = np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8)
            render = lambda item, index: ({"frame": index}, {"depth.png": (depth * 65535).astype(np.uint16),
                                                             "depth.npy": depth})
            write_run(Path(directory), [None], render, 30, "depth.jsonl")
            np.testing.assert_array_equal(np.load(Path(directory, "depth.npy")), depth)
            self.assertTrue(Path(directory, "depth.png").is_file())


if __name__ == "__main__":
    unittest.main()
