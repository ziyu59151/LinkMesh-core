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

PUMI is a site dependency and is not included in the Python environment
created by this repository. The deployment boundary is:

- The user or site administrator provides a compatible PUMI 2.2.9
  installation, the MPI environment, and the matching module/environment
  setup.
- LinkMesh discovers that installation through explicit environment
  variables, compiles the small native helper, installs it into the active
  LinkMesh environment, and verifies its shared libraries.
- LinkMesh does not download, compile, or install PUMI automatically.

A complete M3D-C1 installation does not guarantee that PUMI is visible in the
current shell. Expose the prefix used by the target M3D-C1 stack rather than
assuming a site-specific path.

The PUMI prefix must contain one of:

```text
<prefix>/lib/cmake/SCOREC/SCORECConfig.cmake
<prefix>/lib64/cmake/SCOREC/SCORECConfig.cmake
```

### Discovery and build

Activate either project environment, then set `PUMI_ROOT` explicitly:

```bash
source scripts/activate-venv.sh
export PUMI_ROOT=/path/to/pumi-2.2.9
./scripts/build-smb-helper.sh
linkmesh-check m3dc1
```

When `PUMI_ROOT` is unset, the script checks prefixes in
`CMAKE_PREFIX_PATH` and then, if `spack` is already on `PATH`, runs the
read-only query:

```bash
spack location -i pumi@2.2.9
```

This discovery order is intentional. The user remains responsible for
selecting the PUMI/MPI combination that matches the rest of the M3D-C1 stack.
No module is loaded and no package is installed by the script.

The default build directory is `<repository>/build/helper`. Set
`LINKMESH_BUILD_DIR` to use another directory and `BUILD_JOBS` to choose the
parallel build count. Set `CMAKE_BIN` if CMake is not available as `cmake` or
if the site module must select a specific version.

The script requires CMake 3.20 or newer. If the MPI wrapper is not on `PATH`,
provide it explicitly. When `I_MPI_ROOT` is set and `MPI_CXX` is unset, the
script tries `mpiicpx`, `mpicxx`, and `mpiicpc` under `I_MPI_ROOT/bin`.
Intel MPI also needs `I_MPI_ROOT` while compiling, and the underlying Intel
compiler must be discoverable on `PATH`:

```bash
PUMI_ROOT=/path/to/pumi \
MPI_CXX=/path/to/mpiicpx \
I_MPI_ROOT=/path/to/intel/mpi \
PATH=/path/to/oneapi/compiler/bin:$PATH \
./scripts/build-smb-helper.sh
```

The helper is installed into the active environment and
`linkmesh-check m3dc1` validates its dynamic dependencies. The helper is
host-specific: rebuild it against the target host's PUMI and MPI installation
rather than copying a helper from another MPI environment.

### Troubleshooting

If the script reports that PUMI was not found, check that the CMake package
exists under the prefix and that `PUMI_ROOT` or `CMAKE_PREFIX_PATH` is exported
in the current shell. A directory containing only `libpumi.a` is not enough;
LinkMesh requires the SCOREC CMake package.

If CMake finds a different SCOREC version, load or point to the exact
`2.2.9` installation. Do not silently substitute another version: the helper
and the M3D-C1 mesh workflow are tested against this version.

If the helper builds but `linkmesh-check m3dc1` reports missing libraries,
verify that the MPI wrapper used for the helper is compatible with the MPI
runtime loaded in the current shell. For Intel MPI, `MPI_CXX` and
`I_MPI_ROOT` must point to the same installation. Rebuild after changing
modules or MPI prefixes.

## Existing dependencies

pip reuses dependencies already installed in the active environment when their
versions satisfy `pyproject.toml`. NumPy installed in a different system or
Conda environment is not used by `.venv` unless that venv was deliberately
created with `--system-site-packages`.
