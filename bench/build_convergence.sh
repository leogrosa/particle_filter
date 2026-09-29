#!/usr/bin/env bash
# Idempotent build for mcl_convergence, unlike build_lut.sh/build.sh's rm-rf-every-time pattern.
# Those are for occasional manual reruns; this one gets invoked before every single run by a
# Python driver script, so it needs to be cheap when nothing changed -- only configure with cmake
# the first time (no CMakeCache.txt yet), then always let make's own dependency tracking decide
# whether there's anything to rebuild.
#
# Usage: build_convergence.sh [glt|rm|rmgpu]   (default: glt)
# The range method is fixed at compile time, one binary per method. rmgpu needs CUDA (Jetson
# only), so it gets its own build dir configured WITH_CUDA=ON -- the CPU dir stays CUDA-free.
set -euo pipefail

METHOD="${1:-glt}"
case "$METHOD" in
	glt)   TARGET=mcl_convergence;       BUILD_DIR=build_convergence;      WITH_CUDA=OFF ;;
	rm)    TARGET=mcl_convergence_rm;    BUILD_DIR=build_convergence;      WITH_CUDA=OFF ;;
	rmgpu) TARGET=mcl_convergence_rmgpu; BUILD_DIR=build_convergence_cuda; WITH_CUDA=ON ;;
	*) echo "unknown method '$METHOD' (want glt, rm or rmgpu)" >&2; exit 1 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

if [ ! -f CMakeCache.txt ]; then
	echo "==> Configuring $BUILD_DIR (WITH_CUDA=$WITH_CUDA)"
	cmake -DWITH_CUDA="$WITH_CUDA" ..
fi

# A generated Makefile only re-runs cmake (on CMakeLists.txt changes) when asked for a target it
# already knows -- so a target added since the last configure fails with "No rule to make target"
# unless the configure check runs first. cmake_check_build_system is that check: a no-op when
# nothing changed.
make cmake_check_build_system

echo "==> Building $TARGET"
make "$TARGET"

echo "==> Done: $BUILD_DIR/bin/$TARGET"
