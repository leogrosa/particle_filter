#!/usr/bin/env bash
# Idempotent build for mcl_convergence, unlike build_lut.sh/build.sh's rm-rf-every-time pattern.
# Those are for occasional manual reruns; this one gets invoked before every single run by a
# Python driver script, so it needs to be cheap when nothing changed -- only configure with cmake
# the first time (no CMakeCache.txt yet), then always let make's own dependency tracking decide
# whether there's anything to rebuild.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p build_convergence
cd build_convergence

if [ ! -f CMakeCache.txt ]; then
	echo "==> Configuring (WITH_CUDA=OFF)"
	cmake -DWITH_CUDA=OFF ..
fi

echo "==> Building mcl_convergence"
make mcl_convergence

echo "==> Done: bin/mcl_convergence"
