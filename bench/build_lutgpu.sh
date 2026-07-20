#!/usr/bin/env bash
# Standalone build for just mcl_bench_lutgpu (GiantLUTCastGPU, the GPU-LUT experiment). Needs
# CUDA. Independent build dir (build_lutgpu/) and its own fresh cmake configure, so this can't be
# blocked by, or block, the RM or LUT-clocks builds.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "==> Removing old build_lutgpu/ directory"
rm -rf build_lutgpu

echo "==> Configuring (WITH_CUDA=ON)"
mkdir build_lutgpu
cd build_lutgpu
cmake -DWITH_CUDA=ON ..

echo "==> Building mcl_bench_lutgpu only"
make mcl_bench_lutgpu

echo "==> Done: bin/mcl_bench_lutgpu"
