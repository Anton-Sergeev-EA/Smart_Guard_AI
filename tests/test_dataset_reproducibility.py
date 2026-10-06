import os
import subprocess
import sys
import unittest
from pathlib import Path


class DatasetReproducibility(unittest.TestCase):
    def test_pixels_are_identical_across_process_hash_seeds(self) -> None:
        root = Path(__file__).resolve().parents[1]
        script = """
import hashlib
from backend.ml.generate_dataset import CLASSES, image_seed, make_image
for label in CLASSES:
    picture = make_image(label, image_seed(label, 0, 7))
    print(label, hashlib.sha256(picture.tobytes()).hexdigest())
"""
        outputs: list[str] = []
        for hash_seed in ("1", "999"):
            result = subprocess.run(
                [sys.executable, "-c", script],
                cwd=root,
                env=dict(os.environ, PYTHONHASHSEED=hash_seed),
                capture_output=True,
                text=True,
                check=True,
                timeout=20,
            )
            outputs.append(result.stdout)
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(len(outputs[0].splitlines()), 7)


if __name__ == "__main__":
    unittest.main()
