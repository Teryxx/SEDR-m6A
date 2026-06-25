#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python code/test_final11_models.py "$@"
