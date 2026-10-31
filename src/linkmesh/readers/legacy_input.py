"""Read legacy M3D-C1 meshgen keyword files for single-region meshing."""
from contextlib import contextmanager
from copy import copy
from pathlib import Path
import shlex
from tempfile import TemporaryDirectory

import numpy as np

from linkmesh.config import ConfigError, UNSET


_ARITIES = {
    'simVer': 1, 'modelType': 1, 'inFile': 1, 'outFile': 1,
    'meshSize': 1, 'useVacuumParams': 1, 'adjustVacuumParams': 1,
    'vacuumParams': 5, 'numVacuumPts': 1, 'vacuumFactor': 1,
}


def read_legacy_input(path):
    """Parse keyword/value lines, allowing quoted paths and ! comments."""
    path = Path(path).expanduser().resolve()
    settings = {}
    for number, line in enumerate(path.read_text().splitlines(), 1):
        lexer = shlex.shlex(line, posix=True)
        lexer.whitespace_split = True
        lexer.commenters = '!'
        try:
            fields = list(lexer)
        except ValueError as exc:
            raise ConfigError(f'{path}:{number}: {exc}') from exc
        if not fields:
            continue
        key, *values = fields
        if key not in _ARITIES:
            raise ConfigError(f'{path}:{number}: unsupported legacy setting {key}')
        if key in settings:
            raise ConfigError(f'{path}:{number}: duplicate setting {key}')
        if len(values) != _ARITIES[key]:
            raise ConfigError(f'{path}:{number}: {key} requires {_ARITIES[key]} value(s)')
        settings[key] = values
    return path, settings


@contextmanager
def legacy_arguments(args):
    """Translate legacy settings; temporary analytic boundary lives through meshing."""
    path, settings = read_legacy_input(args.legacy_input)
    def scalar(key, default=None):
        return settings.get(key, [default])[0]
    def integer(key, default):
        try:
            return int(scalar(key, default))
        except (ValueError, TypeError) as exc:
            raise ConfigError(f'{path}: {key} must be an integer') from exc
    model_type = integer('modelType', 0)
    if model_type not in (0, 1):
        raise ConfigError('Legacy input supports modelType=0 or 1 only')
    if integer('adjustVacuumParams', 0) != 0:
        raise ConfigError('adjustVacuumParams must be 0; coordinate scaling is unsupported')
    if integer('useVacuumParams', 0) not in (0, 1):
        raise ConfigError('useVacuumParams must be 0 or 1')
    try:
        dx = float(scalar('meshSize', '0.05'))
    except ValueError as exc:
        raise ConfigError('meshSize must be a positive finite number') from exc
    if not np.isfinite(dx) or dx <= 0:
        raise ConfigError('meshSize must be a positive finite number')
    if args.boundary_source is not UNSET or args.array_file is not UNSET:
        raise ConfigError('--input cannot be combined with --boundary-source or --array-file')
    translated = copy(args)
    translated.legacy_input = None
    translated.no_config = True
    translated.boundary_source = 'array_file'
    name = scalar('outFile', 'diiid')
    for key, value in [('target', 'm3dc1'), ('mesh_dx', dx),
                       ('model', name + '.txt'), ('smb', name + '.smb')]:
        if getattr(translated, key) is UNSET:
            setattr(translated, key, value)
    with TemporaryDirectory(prefix='linkmesh-legacy-') as temporary:
        if model_type == 1:
            filename = scalar('inFile')
            if not filename:
                raise ConfigError('modelType=1 requires inFile')
            translated.array_file = str((path.parent / filename).resolve())
        else:
            if 'vacuumParams' not in settings:
                raise ConfigError('modelType=0 requires five vacuumParams')
            try:
                a, b, c, d, e = map(float, settings['vacuumParams'])
            except ValueError as exc:
                raise ConfigError('vacuumParams must be numeric') from exc
            if not np.isfinite([a, b, c, d, e]).all() or b <= 0 or e <= 0:
                raise ConfigError('vacuumParams must be finite, with positive radial/vertical radii')
            count = integer('numVacuumPts', 20)
            if count < 3:
                raise ConfigError('numVacuumPts must be at least 3')
            # Dense analytic polygon follows the previously tested translation.
            theta = np.linspace(0, 2*np.pi, max(400, count), endpoint=False)
            points = np.column_stack((a+b*np.cos(theta+c*np.sin(theta)), d+e*np.sin(theta)))
            boundary = Path(temporary) / 'boundary.txt'
            np.savetxt(boundary, points, header=str(len(points)), comments='', fmt='%.16e')
            translated.array_file = str(boundary)
        if args.verbose:
            print(f'[LinkMesh] Legacy input: {path}; modelType={model_type}')
        yield translated
