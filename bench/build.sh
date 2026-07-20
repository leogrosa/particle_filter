#!/usr/bin/env bash
# Clean rebuild of bench/'s CMake project with CUDA enabled (Jetson Nano / TX1, sm_53 -- see
# bench/CMakeLists.txt). Wipes build/ first so stale CMake cache / object files from a previous
# WITH_CUDA=OFF configure (or a different toolchain) can't linger.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "==> Removing old build/ directory"
rm -rf build

echo "==> Configuring (WITH_CUDA=ON)"
mkdir build
cd build
cmake -DWITH_CUDA=ON ..

echo "==> Building"
make -j"$(nproc)"

echo "==> Done. Binaries in bench/build/bin/:"
ls -1 bin/
