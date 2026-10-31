"""Legacy parser validation and translation checks (no native dependencies)."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from linkmesh.cli import build_parser
from linkmesh.config import ConfigError
from linkmesh.readers.legacy_input import legacy_arguments


class LegacyInputTests(unittest.TestCase):
    def test_relative_boundary_and_explicit_overrides(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'input'
            path.write_text('modelType 1 ! comment\ninFile "boundary file.txt"\noutFile case\nmeshSize 0.06\n')
            args = build_parser().parse_args(['--input', str(path), '--mesh-dx', '0.02'])
            with legacy_arguments(args) as translated:
                self.assertEqual(translated.array_file, str(Path(directory) / 'boundary file.txt'))
                self.assertEqual(translated.mesh_dx, 0.02)
                self.assertEqual(translated.target, 'm3dc1')
                self.assertEqual(translated.model, 'case.txt')
                self.assertTrue(translated.no_config)

    def test_invalid_settings_are_rejected(self):
        for text in ('modelType 3', 'modelType 1\nadjustVacuumParams 1',
                     'modelType 1\nmeshSize nan', 'modelType 1\nmeshSize -1',
                     'modelType 1\nmodelType 0', 'modelType 1\nunknown 2',
                     'modelType 0\nvacuumParams 1 2'):
            with self.subTest(text=text), TemporaryDirectory() as directory:
                path = Path(directory) / 'input'
                path.write_text(text)
                args = build_parser().parse_args(['--input', str(path)])
                with self.assertRaises(ConfigError), legacy_arguments(args):
                    pass


if __name__ == '__main__':
    unittest.main()
