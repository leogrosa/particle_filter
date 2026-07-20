#!/usr/bin/env bash
# Standalone build for just mcl_bench_lut with the discretize_theta-vs-memory-access clock split
# (LUT_CLOCKS=ON). No CUDA needed. Independent build dir (build_lutclocks/) and its own fresh
# cmake configure, so this can't be blocked by, or block, the RM or GPU-LUT builds.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "==> Removing old build_lutclocks/ directory"
rm -rf build_lutclocks

echo "==> Configuring (WITH_CUDA=OFF, LUT_CLOCKS=ON)"
mkdir build_lutclocks
cd build_lutclocks
cmake -DWITH_CUDA=OFF -DLUT_CLOCKS=ON ..

echo "==> Building mcl_bench_lut only"
make mcl_bench_lut

echo "==> Done: bin/mcl_bench_lut"
