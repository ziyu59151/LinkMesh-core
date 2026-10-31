"""Parser checks for M3D-C1 m3dc1_mfmgen input translation."""
from pathlib import Path
import os
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from linkmesh.config import ConfigError
from linkmesh.readers.mfmgen_input import _offset, parse_mfmgen_input


class MfmgenInputTests(unittest.TestCase):
    def test_tutorial_style_input_and_counted_boundary(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'wall.pts').write_text('4\n1 -1\n2 -1\n2 1\n1 1\n')
            (root / 'input').write_text('''numBdry 1
bdryFile wall.pts 1 0.04
thickWall 1 1 2 0.02
useVacuum 2 3 0.1
vacuumParams 1.5 1.0 0.0 0.0 2.0
numVacuumPts 32
numFace 3
faceBdry 1 1 0.04
faceBdry 2 1 2 0.04
faceBdry 2 2 3 0.1
outFile sample
''')
            previous = Path.cwd()
            try:
                os.chdir(root)
                parsed = parse_mfmgen_input(root / 'input')
            finally:
                os.chdir(previous)
            self.assertEqual(set(parsed['boundaries']), {1, 2, 3})
            self.assertEqual(parsed['out'], 'sample')
            self.assertEqual([face[0] for face in parsed['faces']], [{1}, {1, 2}, {2, 3}])
            self.assertGreater(len(parsed['boundaries'][1]), 4)

    def test_offset_expands_ccw_polygon(self):
        square = np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.]])
        expanded = _offset(square, 0.1)
        self.assertTrue(np.allclose(expanded.min(axis=0), [-0.1, -0.1]))
        self.assertTrue(np.allclose(expanded.max(axis=0), [1.1, 1.1]))

    def test_bad_face_boundary_is_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'wall.pts').write_text('3\n1 0\n2 0\n1 1\n')
            (root / 'input').write_text('''numBdry 1
bdryFile wall.pts 1 0.1
useVacuum 0 0 0.1
numFace 1
faceBdry 2 1 2 0.1
''')
            previous = Path.cwd()
            try:
                os.chdir(root)
                with self.assertRaises(ConfigError):
                    parse_mfmgen_input(root / 'input')
            finally:
                os.chdir(previous)


if __name__ == '__main__':
    unittest.main()
