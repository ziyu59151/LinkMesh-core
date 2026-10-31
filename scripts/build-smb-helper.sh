#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${PUMI_ROOT:-}" ]]; then
  printf 'Set PUMI_ROOT to a compatible PUMI installation prefix.\n' >&2
  exit 2
fi

# shellcheck source=/dev/null
source "$repo_root/scripts/activate-linkmesh.sh"

cmake_bin="${CMAKE_BIN:-cmake}"
mpi_cxx="${MPI_CXX:-mpicxx}"
build_dir="${LINKMESH_BUILD_DIR:-$repo_root/build/helper}"

"$cmake_bin" -S "$repo_root/native/linkmesh_scorec_helper" -B "$build_dir" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="$mpi_cxx" \
  -DCMAKE_PREFIX_PATH="$PUMI_ROOT" \
  -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" \
  -DCMAKE_INSTALL_RPATH="$PUMI_ROOT/lib"
"$cmake_bin" --build "$build_dir" -j "${BUILD_JOBS:-4}"
"$cmake_bin" --install "$build_dir"

export LINKMESH_SCOREC_HELPER="$CONDA_PREFIX/bin/linkmesh-scorec-helper"
linkmesh-check m3dc1
