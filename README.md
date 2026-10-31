# LinkMesh

LinkMesh generates two-dimensional unstructured triangular meshes in tokamak
\(R\)-\(Z\) coordinates. It writes TRIMEG HDF5 files, M3D-C1 model/SMB files,
or mesh-only diagnostic plots.

This repository contains the clean publishable source: Python source, the
native PUMI helper, tests, one example, and the LinkMesh usage manual.
It intentionally excludes AI planning documents, investigation notes,
generated meshes, binaries, and binary release packages.

## Installation

LinkMesh uses `pyproject.toml` for dependencies and one named Conda
environment managed by the project scripts:

```bash
git clone https://github.com/ziyu59151/LinkMesh-core.git
cd LinkMesh-core
./scripts/bootstrap-env.sh
source scripts/activate-linkmesh.sh
linkmesh-check core
```

Set a compatible `PUMI_ROOT` to build and verify the native helper:

```bash
PUMI_ROOT=/path/to/pumi ./scripts/build-smb-helper.sh
```

The required OpenFUSIONToolkit version is pinned to `26.6`. Do not place an
OpenFUSIONToolkit source checkout on `PYTHONPATH`.

Native M3D-C1 SMB output additionally needs SCOREC/PUMI 2.2.9 and a compatible
MPI implementation. `scripts/build-smb-helper.sh` installs the helper into the
active LinkMesh environment and runs `linkmesh-check m3dc1`.

## Usage

Basic mesh-only generation:

```bash
linkmesh --no-config --target mesh_only \
  --boundary-source array_file --array-file boundary.txt \
  --mesh-dx 0.04 --plot mesh.png --quality-plot quality.png
```

TRIMEG HDF5 generation:

```bash
linkmesh --no-config --target trimeg \
  --boundary-source array_file --array-file boundary.txt \
  --mesh-dx 0.04 --hdf5 mesh.h5
```

M3D-C1 model and native SMB generation:

```bash
linkmesh --no-config --target m3dc1 \
  --boundary-source array_file --array-file boundary.txt \
  --mesh-dx 0.04 --model model.txt --smb mesh.smb
```

See [LINKMESH.md](LINKMESH.md) for boundary formats, GFile input, refinement,
legacy meshgen input, mfmgen multi-region input, quality checks, partitioning,
and downstream M3D-C1 limitations.

## Tests

```bash
python -m pip install -e '.[all,dev]'
pytest
```

The automated suite covers input contracts, mesh construction, output formats,
environment checks, and CLI behavior. It does not run the M3D-C1 solver or
validate scientific convergence.

## Format compatibility

The project generates the M3D-C1 model text and native SMB interchange files.
Compatibility is tested with synthetic meshes and local integration cases.
Internal solver source is not included.

The original release materials did not include a LinkMesh-specific license.
External dependencies retain their own licenses.
