#!/usr/bin/env bash
# Standalone build for just mcl_bench (bl/rm/cddt/pcddt/glt CPU sweep, incl. RM). No CUDA needed.
# Independent build dir (build_rm/) and its own fresh cmake configure, so this can't be blocked
# by, or block, the LUT-clocks or GPU-LUT builds.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "==> Removing old build_rm/ directory"
rm -rf build_rm

echo "==> Configuring (WITH_CUDA=OFF)"
mkdir build_rm
cd build_rm
cmake -DWITH_CUDA=OFF ..

echo "==> Building mcl_bench only"
make mcl_bench

echo "==> Done: bin/mcl_bench"
