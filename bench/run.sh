#!/usr/bin/env bash
# Thin wrapper: runs the RMGPU benchmark multiple times (warmup + varying seeds) and reports
# statistics. All the actual logic lives in bench_stats.py (a generalized version of the old
# rmgpu_stats.py that also works against mcl_bench / mcl_bench_lut) -- run `./run.sh --help`
# for options, e.g. `./run.sh --repeats 10 --hz-targets 40,20,10` or `./run.sh --output-csv raw.csv`.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BINARY="build/bin/mcl_bench_rmgpu"
if [[ ! -x "$BINARY" ]]; then
    echo "error: $BINARY not found or not executable. Run ./build.sh first." >&2
    exit 1
fi

python3 bench_stats.py --binary "$BINARY" "$@"
