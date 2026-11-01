#!/usr/bin/env bash

_linkmesh_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_venv_dir="${LINKMESH_VENV:-$_linkmesh_root/.venv}"

if [[ ! -f "$_venv_dir/bin/activate" ]]; then
  printf 'Virtual environment not found: %s\n' "$_venv_dir" >&2
  printf 'Run %s/scripts/bootstrap-venv.sh first.\n' "$_linkmesh_root" >&2
  return 1 2>/dev/null || exit 1
fi

# shellcheck source=/dev/null
source "$_venv_dir/bin/activate"

export LINKMESH_ROOT="$_linkmesh_root"
if [[ -x "$VIRTUAL_ENV/bin/linkmesh-scorec-helper" ]]; then
  export LINKMESH_SCOREC_HELPER="$VIRTUAL_ENV/bin/linkmesh-scorec-helper"
else
  unset LINKMESH_SCOREC_HELPER
fi

unset _linkmesh_root _venv_dir
