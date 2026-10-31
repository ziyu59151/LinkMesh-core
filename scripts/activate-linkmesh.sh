#!/usr/bin/env bash

_linkmesh_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_conda_base="${CONDA_BASE:-$HOME/miniconda3}"

if [[ ! -f "$_conda_base/etc/profile.d/conda.sh" ]]; then
  printf 'Conda initialization not found: %s\n' "$_conda_base" >&2
  return 1 2>/dev/null || exit 1
fi

# shellcheck source=/dev/null
source "$_conda_base/etc/profile.d/conda.sh"
conda activate linkmesh

export LINKMESH_ROOT="$_linkmesh_root"
if [[ -x "$CONDA_PREFIX/bin/linkmesh-scorec-helper" ]]; then
  export LINKMESH_SCOREC_HELPER="$CONDA_PREFIX/bin/linkmesh-scorec-helper"
fi

if command -v nc >/dev/null 2>&1 && nc -z -w 1 127.0.0.1 17897 2>/dev/null; then
  export LINKMESH_GITHUB_PROXY="http://127.0.0.1:17897"
fi

unset _linkmesh_root _conda_base
