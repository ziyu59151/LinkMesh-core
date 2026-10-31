"""Dependency checks for installed LinkMesh packages and native SMB helpers."""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import os
from pathlib import Path
import shutil
import subprocess
import sys


def _status(label: str, message: str) -> None:
    print(f'{label:<12} {message}')


def _module(name: str) -> bool:
    try:
        module = importlib.import_module(name)
        if name == 'matplotlib':
            module.use('Agg', force=True)
            importlib.import_module('matplotlib.pyplot')
        _status('OK', f'{name}: {getattr(module, "__file__", "built-in")}')
        return True
    except Exception as exc:
        _status('MISSING', f'{name}: {exc}')
        return False


def _core() -> bool:
    _status('OK', f'Python: {sys.executable} ({sys.version.split()[0]})')
    results = [_module(name) for name in ('linkmesh', 'numpy', 'OpenFUSIONToolkit')]
    try:
        version = importlib.metadata.version('openfusiontoolkit')
        valid = version == '26.6'
        _status('OK' if valid else 'MISSING', f'OpenFUSIONToolkit: {version}; required 26.6')
        results.append(valid)
        from OpenFUSIONToolkit.TokaMaker.meshing import gs_Domain
        _status('OK', f'Native meshing interface: {gs_Domain.__name__}')
    except Exception as exc:
        _status('MISSING', f'OpenFUSIONToolkit native meshing: {exc}')
        results.append(False)
    return all(results)


def _native() -> bool:
    candidate = os.environ.get('LINKMESH_SCOREC_HELPER') or shutil.which('linkmesh-scorec-helper')
    if not candidate or not Path(candidate).is_file() or not os.access(candidate, os.X_OK):
        _status('MISSING', 'Executable SCOREC helper; set LINKMESH_SCOREC_HELPER')
        return False
    helper = str(Path(candidate).resolve())
    _status('OK', f'SCOREC helper: {helper}')
    if not shutil.which('ldd'):
        _status('MISSING', 'ldd unavailable; native dependency validation requires Linux')
        return False
    result = subprocess.run(['ldd', helper], capture_output=True, text=True, timeout=30)
    output = result.stdout + result.stderr
    valid = result.returncode == 0 and 'not found' not in output
    _status('OK' if valid else 'MISSING', 'Native dynamic-library resolution')
    print(output.rstrip())
    mpi = any(line.lstrip().startswith('libmpi.so') and 'not found' not in line for line in output.splitlines())
    _status('OK' if mpi else 'MISSING', 'MPI runtime dependency')
    _status('NOT CHECKED', 'SMB generation, partitioning, M3D-C1 initialization and time stepping')
    return valid and mpi


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('core', 'plot', 'trimeg', 'm3dc1', 'all'))
    args = parser.parse_args()
    modes = ('core', 'plot', 'trimeg', 'm3dc1') if args.mode == 'all' else (args.mode,)
    results = []
    for mode in modes:
        print(f'\n== {mode} ==')
        valid = _core()
        if mode == 'plot':
            valid = _module('matplotlib') and valid
        elif mode == 'trimeg':
            valid = _module('h5py') and valid
        elif mode == 'm3dc1':
            valid = _native() and valid
        results.append(valid)
    raise SystemExit(0 if all(results) else 1)
