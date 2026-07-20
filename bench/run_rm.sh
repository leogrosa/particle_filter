#!/usr/bin/env bash
# Thin wrapper around bench_stats.py --points-mode for mcl_bench (bl/rm/cddt/pcddt/glt CPU
# sweep). Unlike run.sh (which hands mcl_bench_rmgpu a seed and lets it run its whole internal
# sweep in one call), mcl_bench now takes --method/--rays/--particles and runs exactly one point
# per invocation -- so bench_stats.py can apply --timeout per particle count and skip the rest of
# a method's sweep once one point times out, instead of one global timeout over everything.
#
# Defaults to the 60-ray sweep only (the 1080-ray sweep is expensive for RM's data-dependent
# marching and usually isn't needed -- pass --rays 1080 --particles ... explicitly to run it).
# Examples:
#   ./run_rm.sh
#   ./run_rm.sh --methods rm --particles 500,1000,2000,4000,8000,11700,16000 --timeout 30
#   ./run_rm.sh --rays 1080 --particles 100,300,650,1000,2000 --timeout 60
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BINARY="build_rm/bin/mcl_bench"
if [[ ! -x "$BINARY" ]]; then
    echo "error: $BINARY not found or not executable. Run ./build_rm.sh first." >&2
    exit 1
fi

DEFAULT_PARTICLES="500,1000,2000,4000,8000,11700,16000,24000,32000,50000,75000,100000"

python3 bench_stats.py --points-mode --binary "$BINARY" \
    --rays 60 --particles "$DEFAULT_PARTICLES" --timeout 30 \
    "$@"
