#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
"$PYTHON" "$ROOT/code/evaluate_thyqc.py" --all-seeds
