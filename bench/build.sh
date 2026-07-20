#!/usr/bin/env bash
# Clean rebuild of bench/'s CMake project with CUDA enabled (Jetson Nano / TX1, sm_53 -- see
# bench/CMakeLists.txt). Wipes build/ first so stale CMake cache / object files from a previous
# WITH_CUDA=OFF configure (or a different toolchain) can't linger.
#
# Optional flags:
#   --lut-clocks  Also configure -DLUT_CLOCKS=ON (mcl_bench_lut's opt-in discretize_theta-vs-
#                 memory-access clock split, see bench/CMakeLists.txt). WITH_CUDA=ON stays on
#                 regardless.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CMAKE_EXTRA_ARGS=()
for arg in "$@"; do
	case "$arg" in
		--lut-clocks)
			CMAKE_EXTRA_ARGS+=("-DLUT_CLOCKS=ON")
			;;
		*)
			echo "unknown argument: $arg" >&2
			exit 1
			;;
	esac
done

echo "==> Removing old build/ directory"
rm -rf build

echo "==> Configuring (WITH_CUDA=ON ${CMAKE_EXTRA_ARGS[*]:-})"
mkdir build
cd build
# ${arr[@]+"${arr[@]}"} (rather than plain "${arr[@]}") avoids an "unbound variable" error
# under `set -u` when CMAKE_EXTRA_ARGS is empty, on older bash (e.g. macOS's default bash 3.2).
cmake -DWITH_CUDA=ON ${CMAKE_EXTRA_ARGS[@]+"${CMAKE_EXTRA_ARGS[@]}"} ..

echo "==> Building"
make -j"$(nproc)"

echo "==> Done. Binaries in bench/build/bin/:"
ls -1 bin/
