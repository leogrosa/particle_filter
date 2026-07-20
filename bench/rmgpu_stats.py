#!/usr/bin/env python3
"""
Runs mcl_bench_rmgpu several times under different seeds and aggregates statistics.

Why repeat runs at all: RM/RMGPU's per-ray marching step count is data-dependent (distance to
the nearest wall along that specific ray), unlike GLT/CDDT's O(1) queries. mcl_bench_rmgpu.cpp
samples particle positions with a single RNG seeded once at program start, so a single run's
sweep can show real (non-noise) non-monotonic timing purely because of which map regions
happened to get sampled at each particle count -- not measurement error. Averaging several
independent seeds separates true particle-count scaling from that per-seed sampling variance.

Stdlib only (subprocess/csv/statistics) -- no numpy/pandas dependency required on the Jetson.
"""
import argparse
import csv
import io
import statistics
import subprocess
import sys
from collections import defaultdict

NOISY_REL_STDEV = 0.15  # flag a point if stdev exceeds this fraction of the mean


def run_once(binary, seed, timeout):
    # stdout/stderr=PIPE + universal_newlines, not capture_output/text -- the Jetson's stock
    # Python is 3.6, and both of those kwargs were only added in 3.7.
    result = subprocess.run(
        [binary, str(seed)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        check=True,
        timeout=timeout,
    )
    return result.stdout


def parse_rows(text):
    rows = []
    for row in csv.DictReader(io.StringIO(text)):
        row["max_particles"] = int(row["max_particles"])
        row["num_rays"] = int(row["num_rays"])
        for k in ("iters_per_sec", "ms_total", "ms_resample", "ms_motion",
                  "ms_range_sensor", "ms_normalize", "ms_gpu_range"):
            if k in row and row[k] != "":
                row[k] = float(row[k])
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary", default="build/bin/mcl_bench_rmgpu", help="path to the mcl_bench_rmgpu executable")
    ap.add_argument("--repeats", type=int, default=5, help="timed repetitions (default 5)")
    ap.add_argument("--warmup-runs", type=int, default=1,
                     help="throwaway full invocations before timed repeats, to let clocks/thermal settle (default 1)")
    ap.add_argument("--seed-start", type=int, default=1000, help="first seed used; repeat i uses seed-start+i")
    ap.add_argument("--hz-targets", default="40,10", help="comma-separated Hz budgets to report sustained particle counts for")
    ap.add_argument("--output-csv", default=None, help="optional path to dump every raw row from every run (long format)")
    ap.add_argument("--timeout", type=float, default=600.0, help="per-invocation timeout in seconds")
    args = ap.parse_args()

    hz_targets = [float(x) for x in args.hz_targets.split(",") if x.strip()]

    print(f"==> {args.warmup_runs} warmup run(s) (discarded, letting clocks/thermal settle)")
    for i in range(args.warmup_runs):
        run_once(args.binary, seed=1, timeout=args.timeout)
        print(f"    warmup {i + 1}/{args.warmup_runs} done")

    print(f"==> {args.repeats} timed run(s), seeds {args.seed_start}..{args.seed_start + args.repeats - 1}")
    all_rows = []
    # grouped[(num_rays, max_particles)][field] -> list of values, one per repeat
    grouped = defaultdict(lambda: defaultdict(list))
    for i in range(args.repeats):
        seed = args.seed_start + i
        out = run_once(args.binary, seed, timeout=args.timeout)
        rows = parse_rows(out)
        for row in rows:
            row["seed"] = seed
            all_rows.append(row)
            key = (row["num_rays"], row["max_particles"])
            for field in ("ms_total", "ms_resample", "ms_motion", "ms_range_sensor",
                          "ms_normalize", "ms_gpu_range", "iters_per_sec"):
                grouped[key][field].append(row[field])
        print(f"    run {i + 1}/{args.repeats} (seed={seed}) done, {len(rows)} rows")

    if not all_rows:
        print("No rows parsed from any run -- check the binary path and its stdout format.", file=sys.stderr)
        sys.exit(1)

    if args.output_csv:
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"==> Raw per-run rows written to {args.output_csv}")

    print()
    print(f"==> Aggregated statistics (across {args.repeats} run(s) per point)")
    header = f"{'rays':>5} {'particles':>10} {'mean_ms':>10} {'stdev_ms':>10} {'min_ms':>10} {'max_ms':>10} {'mean_hz':>9}"
    print(header)
    print("-" * len(header))

    summary = defaultdict(list)  # num_rays -> list of (particles, mean_ms, stdev_ms)
    for (num_rays, particles), fields in sorted(grouped.items()):
        vals = fields["ms_total"]
        mean_ms = statistics.mean(vals)
        stdev_ms = statistics.stdev(vals) if len(vals) > 1 else 0.0
        min_ms, max_ms = min(vals), max(vals)
        mean_hz = 1000.0 / mean_ms
        flag = " *" if mean_ms > 0 and (stdev_ms / mean_ms) > NOISY_REL_STDEV else ""
        print(f"{num_rays:>5} {particles:>10} {mean_ms:>10.3f} {stdev_ms:>10.3f} "
              f"{min_ms:>10.3f} {max_ms:>10.3f} {mean_hz:>9.2f}{flag}")
        summary[num_rays].append((particles, mean_ms, stdev_ms))
    print(f"(* = stdev exceeds {NOISY_REL_STDEV:.0%} of the mean -- treat that point cautiously)")

    print()
    print("==> Sustained particle counts at target frequencies (based on mean ms_total)")
    for num_rays, points in sorted(summary.items()):
        points.sort()  # by particle count, ascending
        print(f"  -- {num_rays} rays/particle --")
        for hz in hz_targets:
            budget_ms = 1000.0 / hz
            # Scan the full sorted list rather than stopping at the first miss: RM/RMGPU can be
            # non-monotonic even after averaging, so "sustained" here means the largest particle
            # count whose mean stays under budget, not the first point that happens to exceed it.
            sustained = None
            for particles, mean_ms, stdev_ms in points:
                if mean_ms <= budget_ms:
                    sustained = (particles, mean_ms, stdev_ms)
            if sustained:
                particles, mean_ms, stdev_ms = sustained
                print(f"    {hz:>5.1f} Hz (budget {budget_ms:.2f}ms): sustained up to {particles} particles "
                      f"(mean {mean_ms:.3f}ms +/- {stdev_ms:.3f}ms)")
            else:
                print(f"    {hz:>5.1f} Hz (budget {budget_ms:.2f}ms): not sustained even at the smallest tested particle count")


if __name__ == "__main__":
    main()
