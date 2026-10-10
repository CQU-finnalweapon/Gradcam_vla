#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
gradcam_python="${GRADCAM_PYTHON:-/mnt/vepfs01/output/klay.zhou/.venvs/mozbrain-gradcam/bin/python}"
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
exec "$gradcam_python" -u "$script_dir/run.py" "$@"
