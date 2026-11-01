#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON:-python3}"
venv_dir="${LINKMESH_VENV:-$repo_root/.venv}"

if ! command -v "$python_bin" >/dev/null 2>&1; then
  printf 'Python executable not found: %s\n' "$python_bin" >&2
  exit 2
fi

"$python_bin" -m venv "$venv_dir"
# shellcheck source=/dev/null
source "$venv_dir/bin/activate"

python -m pip install --upgrade pip
python -m pip install -e "${repo_root}[all,dev]"
python -m pip check
linkmesh-check core

printf 'LinkMesh virtual environment ready: %s\n' "$VIRTUAL_ENV"
printf 'Activate later with: source %s/scripts/activate-venv.sh\n' "$repo_root"
