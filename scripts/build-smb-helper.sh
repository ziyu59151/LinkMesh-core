#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${PUMI_ROOT:-}" ]]; then
  printf 'Set PUMI_ROOT to a compatible PUMI installation prefix.\n' >&2
  exit 2
fi

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  env_prefix="$VIRTUAL_ENV"
elif [[ "${CONDA_DEFAULT_ENV:-}" == "linkmesh" && -n "${CONDA_PREFIX:-}" ]]; then
  env_prefix="$CONDA_PREFIX"
else
  printf 'Activate the project venv or the named linkmesh Conda environment first.\n' >&2
  exit 2
fi

if ! command -v linkmesh-check >/dev/null 2>&1; then
  printf 'linkmesh-check is not available in the active environment.\n' >&2
  printf 'Run the corresponding bootstrap script first.\n' >&2
  exit 2
fi

cmake_bin="${CMAKE_BIN:-cmake}"
mpi_cxx="${MPI_CXX:-mpicxx}"
build_dir="${LINKMESH_BUILD_DIR:-$repo_root/build/helper}"

if ! command -v "$cmake_bin" >/dev/null 2>&1; then
  printf 'CMake executable not found: %s\n' "$cmake_bin" >&2
  exit 2
fi
if [[ ! -d "$PUMI_ROOT" ]]; then
  printf 'PUMI_ROOT is not a directory: %s\n' "$PUMI_ROOT" >&2
  exit 2
fi
if ! command -v "$mpi_cxx" >/dev/null 2>&1 && [[ ! -x "$mpi_cxx" ]]; then
  printf 'MPI C++ compiler not found: %s\n' "$mpi_cxx" >&2
  exit 2
fi

"$cmake_bin" -S "$repo_root/native/linkmesh_scorec_helper" -B "$build_dir" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="$mpi_cxx" \
  -DCMAKE_PREFIX_PATH="$PUMI_ROOT" \
  -DCMAKE_INSTALL_PREFIX="$env_prefix" \
  -DCMAKE_INSTALL_RPATH="$PUMI_ROOT/lib"
"$cmake_bin" --build "$build_dir" -j "${BUILD_JOBS:-4}"
"$cmake_bin" --install "$build_dir"

export LINKMESH_SCOREC_HELPER="$env_prefix/bin/linkmesh-scorec-helper"
linkmesh-check m3dc1
