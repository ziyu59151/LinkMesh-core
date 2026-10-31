#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
conda_base="${CONDA_BASE:-$HOME/miniconda3}"

# shellcheck source=/dev/null
source "$conda_base/etc/profile.d/conda.sh"

if ! conda env list | awk '$1 == "linkmesh" { found = 1 } END { exit !found }'; then
  conda create -y -n linkmesh python=3.12 pip
fi

conda activate linkmesh
python -m pip install --upgrade pip
python -m pip install -e "${repo_root}[all,dev]"
python -m pip check

printf 'LinkMesh environment ready: %s\n' "$CONDA_PREFIX"
