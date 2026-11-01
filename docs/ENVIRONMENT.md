# Environment setup

LinkMesh uses `pyproject.toml` to declare Python dependencies. The virtual
environment is the default deployment path; Conda is optional for users who
already manage HPC software that way.

## Virtual environment

Create and verify the environment:

```bash
./scripts/bootstrap-venv.sh
source scripts/activate-venv.sh
linkmesh-check core
```

The environment is created at `.venv/` and its packages are installed under
`.venv/lib/pythonX.Y/site-packages/`. It does not modify the system Python,
Conda base, or other projects. Set `PYTHON=/path/to/python3` to choose the
interpreter and `LINKMESH_VENV=/path/to/venv` to use another location.

## Conda

On hosts that already provide Conda:

```bash
./scripts/bootstrap-conda.sh
source scripts/activate-conda.sh
linkmesh-check core
```

The script reuses or creates the named `linkmesh` environment. It records
dependencies in that environment only. Set `CONDA_BASE` if Miniconda is not at
`$HOME/miniconda3`.

## Native SMB helper

The Python mesh, PNG, TRIMEG HDF5, and model TXT paths need no native helper.
M3D-C1 SMB output additionally requires CMake, a C++17 MPI compiler, and
SCOREC/PUMI 2.2.9.

Activate either environment, then:

```bash
PUMI_ROOT=/path/to/pumi ./scripts/build-smb-helper.sh
```

If the MPI wrapper is not on `PATH`, provide it explicitly. Intel MPI also
uses `I_MPI_ROOT` so its compiler wrapper can locate the installation:

```bash
PUMI_ROOT=/path/to/pumi \
MPI_CXX=/path/to/mpiicpx \
I_MPI_ROOT=/path/to/intel/mpi \
./scripts/build-smb-helper.sh
```

The helper is installed into the active environment and
`linkmesh-check m3dc1` validates its dynamic dependencies.

## Existing dependencies

pip reuses dependencies already installed in the active environment when their
versions satisfy `pyproject.toml`. NumPy installed in a different system or
Conda environment is not used by `.venv` unless that venv was deliberately
created with `--system-site-packages`.
