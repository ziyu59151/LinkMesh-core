#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

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
build_dir="${LINKMESH_BUILD_DIR:-$repo_root/build/helper}"
pumi_root="${PUMI_ROOT:-}"
extra_prefixes="${CMAKE_PREFIX_PATH:-}"

if ! command -v "$cmake_bin" >/dev/null 2>&1; then
  printf 'CMake executable not found: %s\n' "$cmake_bin" >&2
  printf 'Load a CMake module or set CMAKE_BIN to a CMake >= 3.20 executable.\n' >&2
  exit 2
fi

cmake_version="$("$cmake_bin" --version | awk 'NR == 1 { print $3 }')"
if [[ ! "$cmake_version" =~ ^([0-9]+)\.([0-9]+) ]]; then
  printf 'Could not parse CMake version from: %s\n' "$cmake_version" >&2
  exit 2
fi
cmake_major="${BASH_REMATCH[1]}"
cmake_minor="${BASH_REMATCH[2]}"
if (( cmake_major < 3 || (cmake_major == 3 && cmake_minor < 20) )); then
  printf 'CMake >= 3.20 is required; found %s (%s).\n' "$cmake_version" "$cmake_bin" >&2
  printf 'Load a newer CMake module or set CMAKE_BIN explicitly.\n' >&2
  exit 2
fi

# PUMI is a site dependency.  Resolve an existing installation through an
# explicit prefix, CMAKE_PREFIX_PATH, or Spack's read-only location query.
if [[ -z "$pumi_root" ]]; then
  IFS=':;' read -r -a prefix_entries <<< "$extra_prefixes"
  for prefix in "${prefix_entries[@]}"; do
    if [[ -f "$prefix/lib/cmake/SCOREC/SCORECConfig.cmake" ]] ||
       [[ -f "$prefix/lib64/cmake/SCOREC/SCORECConfig.cmake" ]]; then
      pumi_root="$prefix"
      break
    fi
  done
fi

if [[ -z "$pumi_root" ]] && command -v spack >/dev/null 2>&1; then
  spack_pumi_root="$(spack location -i pumi@2.2.9 2>/dev/null || true)"
  if [[ -n "$spack_pumi_root" ]]; then
    pumi_root="$spack_pumi_root"
  fi
fi

if [[ -z "$pumi_root" ]]; then
  printf 'PUMI was not found. LinkMesh does not install PUMI automatically.\n' >&2
  printf 'Set PUMI_ROOT to a PUMI 2.2.9 prefix, or add that prefix to CMAKE_PREFIX_PATH.\n' >&2
  printf 'Expected one of:\n' >&2
  printf '  <prefix>/lib/cmake/SCOREC/SCORECConfig.cmake\n' >&2
  printf '  <prefix>/lib64/cmake/SCOREC/SCORECConfig.cmake\n' >&2
  exit 2
fi

if [[ ! -d "$pumi_root" ]]; then
  printf 'PUMI_ROOT is not a directory: %s\n' "$pumi_root" >&2
  exit 2
fi

if [[ -f "$pumi_root/lib/cmake/SCOREC/SCORECConfig.cmake" ]]; then
  scorec_dir="$pumi_root/lib/cmake/SCOREC"
  pumi_lib_dir="$pumi_root/lib"
elif [[ -f "$pumi_root/lib64/cmake/SCOREC/SCORECConfig.cmake" ]]; then
  scorec_dir="$pumi_root/lib64/cmake/SCOREC"
  pumi_lib_dir="$pumi_root/lib64"
else
  printf 'PUMI configuration was not found under: %s\n' "$pumi_root" >&2
  printf 'LinkMesh requires the SCOREC CMake package, not just a PUMI library directory.\n' >&2
  exit 2
fi

if [[ -n "${MPI_CXX:-}" ]]; then
  mpi_cxx="$MPI_CXX"
elif [[ -n "${I_MPI_ROOT:-}" ]]; then
  mpi_cxx=""
  for candidate in "$I_MPI_ROOT/bin/mpiicpx" "$I_MPI_ROOT/bin/mpicxx" "$I_MPI_ROOT/bin/mpiicpc"; do
    if [[ -x "$candidate" ]]; then
      mpi_cxx="$candidate"
      break
    fi
  done
  if [[ -z "$mpi_cxx" ]]; then
    mpi_cxx="mpicxx"
  fi
else
  mpi_cxx="mpicxx"
fi

if [[ "$mpi_cxx" == */* ]]; then
  mpi_available=0
  [[ -x "$mpi_cxx" ]] && mpi_available=1
else
  mpi_available=0
  command -v "$mpi_cxx" >/dev/null 2>&1 && mpi_available=1
fi
if (( mpi_available == 0 )); then
  printf 'MPI C++ compiler not found: %s\n' "$mpi_cxx" >&2
  printf 'Load the MPI module, put its wrapper on PATH, or set MPI_CXX explicitly.\n' >&2
  printf 'For Intel MPI, also set I_MPI_ROOT when required by the wrapper.\n' >&2
  exit 2
fi

if ! mpi_version_output="$("$mpi_cxx" --version 2>&1)"; then
  printf 'The MPI C++ wrapper could not start: %s\n' "$mpi_cxx" >&2
  printf 'Load the matching compiler module as well as MPI. For mpiicpx, icpx must be on PATH.\n' >&2
  printf '%s\n' "$mpi_version_output" | sed -n '1,8p' >&2
  exit 2
fi

if [[ -n "$extra_prefixes" ]]; then
  # CMake accepts semicolon-separated prefixes, while the environment form
  # commonly uses colons on Unix.
  extra_prefixes="${extra_prefixes//:/;}"
  cmake_prefix_path="$pumi_root;$extra_prefixes"
else
  cmake_prefix_path="$pumi_root"
fi

"$cmake_bin" -S "$repo_root/native/linkmesh_scorec_helper" -B "$build_dir" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="$mpi_cxx" \
  -DCMAKE_PREFIX_PATH="$cmake_prefix_path" \
  -DSCOREC_DIR="$scorec_dir" \
  -DCMAKE_INSTALL_PREFIX="$env_prefix" \
  -DCMAKE_INSTALL_RPATH="$pumi_lib_dir"
"$cmake_bin" --build "$build_dir" -j "${BUILD_JOBS:-4}"
"$cmake_bin" --install "$build_dir"

export LINKMESH_SCOREC_HELPER="$env_prefix/bin/linkmesh-scorec-helper"
linkmesh-check m3dc1
