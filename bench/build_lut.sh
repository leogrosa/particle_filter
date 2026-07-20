#!/usr/bin/env bash
# Standalone build for just mcl_bench_lut, WITHOUT the LUT_CLOCKS instrumentation (plain glt
# sweep only -- no discretize_theta-vs-memory-access clock split). No CUDA needed. Independent
# build dir (build_lut/) so this can't collide with build_lutclocks/'s LUT_CLOCKS=ON build.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "==> Removing old build_lut/ directory"
rm -rf build_lut

echo "==> Configuring (WITH_CUDA=OFF, LUT_CLOCKS=OFF)"
mkdir build_lut
cd build_lut
cmake -DWITH_CUDA=OFF ..

echo "==> Building mcl_bench_lut only"
make mcl_bench_lut

echo "==> Done: bin/mcl_bench_lut"
