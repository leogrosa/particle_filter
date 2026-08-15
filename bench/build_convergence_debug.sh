#!/usr/bin/env bash
# Debug build of mcl_convergence (-g -O0, DEBUG_CONVERGENCE=ON) in its own build dir so it never
# clobbers the Release build in build_convergence/. Idempotent like build_convergence.sh: only
# configures with cmake the first time, then lets make's own dependency tracking decide what to
# rebuild.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p build_convergence_debug
cd build_convergence_debug

if [ ! -f CMakeCache.txt ]; then
	echo "==> Configuring (WITH_CUDA=OFF, DEBUG_CONVERGENCE=ON)"
	cmake -DWITH_CUDA=OFF -DDEBUG_CONVERGENCE=ON ..
fi

echo "==> Building mcl_convergence (debug)"
make mcl_convergence

echo "==> Done: bin/mcl_convergence (debug)"
