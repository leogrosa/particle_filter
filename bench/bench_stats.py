#!/usr/bin/env python3
"""
Runs one of the mcl_bench* CSV-emitting binaries several times under different seeds and
aggregates statistics. Works against mcl_bench, mcl_bench_lut, and mcl_bench_rmgpu -- they all
share the same CSV columns (method,max_particles,num_rays,iters_per_sec,ms_total,ms_resample,
ms_motion,ms_range_sensor,ms_normalize), with mcl_bench_rmgpu alone adding a trailing
ms_gpu_range column.

Why repeat runs at all: RM/RMGPU's per-ray marching step count is data-dependent (distance to
the nearest wall along that specific ray), unlike GLT/CDDT's O(1) queries. The binaries sample
particle positions with a single RNG seeded once at program start, so a single run's sweep can
show real (non-noise) non-monotonic timing purely because of which map regions happened to get
sampled at each particle count -- not measurement error. Averaging several independent seeds
separates true particle-count scaling from that per-seed sampling variance. (bl/cddt/pcddt/glt
are effectively O(1)-per-ray and don't need this, but running them through the same averaging
does no harm.)

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

# Present in every binary's CSV output. ms_gpu_range is optional (mcl_bench_rmgpu only) and is
# handled separately wherever it matters.
CORE_FIELDS = ("ms_total", "ms_resample", "ms_motion", "ms_range_sensor", "ms_normalize", "iters_per_sec")
OPTIONAL_FIELDS = ("ms_gpu_range",)


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


def parse_rows(text, method_filter=None):
    rows = []
    for row in csv.DictReader(io.StringIO(text)):
        if method_filter and row["method"] not in method_filter:
            continue
        row["max_particles"] = int(row["max_particles"])
        row["num_rays"] = int(row["num_rays"])
        for k in CORE_FIELDS + OPTIONAL_FIELDS:
            if k in row and row[k] != "":
                row[k] = float(row[k])
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary", default="build/bin/mcl_bench_rmgpu",
                     help="path to the mcl_bench / mcl_bench_lut / mcl_bench_rmgpu executable")
    ap.add_argument("--repeats", type=int, default=5, help="timed repetitions (default 5)")
    ap.add_argument("--warmup-runs", type=int, default=1,
                     help="throwaway full invocations before timed repeats, to let clocks/thermal settle (default 1)")
    ap.add_argument("--seed-start", type=int, default=1000, help="first seed used; repeat i uses seed-start+i")
    ap.add_argument("--hz-targets", default="40,10", help="comma-separated Hz budgets to report sustained particle counts for")
    ap.add_argument("--output-csv", default=None, help="optional path to dump every raw row from every run (long format)")
    ap.add_argument("--timeout", type=float, default=600.0, help="per-invocation timeout in seconds")
    ap.add_argument("--method-filter", default=None,
                     help="comma-separated method names to keep (e.g. 'rm' or 'rm,glt'); "
                          "default: no filtering, keep every method the binary emits")
    args = ap.parse_args()

    hz_targets = [float(x) for x in args.hz_targets.split(",") if x.strip()]
    method_filter = None
    if args.method_filter:
        method_filter = {m.strip() for m in args.method_filter.split(",") if m.strip()}

    print(f"==> {args.warmup_runs} warmup run(s) (discarded, letting clocks/thermal settle)")
    for i in range(args.warmup_runs):
        run_once(args.binary, seed=1, timeout=args.timeout)
        print(f"    warmup {i + 1}/{args.warmup_runs} done")

    print(f"==> {args.repeats} timed run(s), seeds {args.seed_start}..{args.seed_start + args.repeats - 1}")
    all_rows = []
    # grouped[(method, num_rays, max_particles)][field] -> list of values, one per repeat
    grouped = defaultdict(lambda: defaultdict(list))
    has_gpu_range = False
    for i in range(args.repeats):
        seed = args.seed_start + i
        out = run_once(args.binary, seed, timeout=args.timeout)
        rows = parse_rows(out, method_filter)
        for row in rows:
            row["seed"] = seed
            all_rows.append(row)
            key = (row["method"], row["num_rays"], row["max_particles"])
            for field in CORE_FIELDS:
                grouped[key][field].append(row[field])
            if "ms_gpu_range" in row:
                has_gpu_range = True
                grouped[key]["ms_gpu_range"].append(row["ms_gpu_range"])
        print(f"    run {i + 1}/{args.repeats} (seed={seed}) done, {len(rows)} rows")

    if not all_rows:
        print("No rows parsed from any run -- check the binary path, its stdout format, and --method-filter.",
              file=sys.stderr)
        sys.exit(1)

    if args.output_csv:
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"==> Raw per-run rows written to {args.output_csv}")

    print()
    print(f"==> Aggregated statistics (across {args.repeats} run(s) per point)")
    header = f"{'method':>8} {'rays':>5} {'particles':>10} {'mean_ms':>10} {'stdev_ms':>10} {'min_ms':>10} {'max_ms':>10} {'mean_hz':>9}"
    print(header)
    print("-" * len(header))

    # summary[(method, num_rays)] -> list of (particles, mean_ms, stdev_ms)
    summary = defaultdict(list)
    for (method, num_rays, particles), fields in sorted(grouped.items()):
        vals = fields["ms_total"]
        mean_ms = statistics.mean(vals)
        stdev_ms = statistics.stdev(vals) if len(vals) > 1 else 0.0
        min_ms, max_ms = min(vals), max(vals)
        mean_hz = 1000.0 / mean_ms
        flag = " *" if mean_ms > 0 and (stdev_ms / mean_ms) > NOISY_REL_STDEV else ""
        print(f"{method:>8} {num_rays:>5} {particles:>10} {mean_ms:>10.3f} {stdev_ms:>10.3f} "
              f"{min_ms:>10.3f} {max_ms:>10.3f} {mean_hz:>9.2f}{flag}")
        summary[(method, num_rays)].append((particles, mean_ms, stdev_ms))
    print(f"(* = stdev exceeds {NOISY_REL_STDEV:.0%} of the mean -- treat that point cautiously)")

    if has_gpu_range:
        print()
        print("==> ms_gpu_range (numpy_calc_range_angles only, sub-split of ms_range_sensor)")
        gpu_header = f"{'method':>8} {'rays':>5} {'particles':>10} {'mean_ms':>10} {'stdev_ms':>10}"
        print(gpu_header)
        print("-" * len(gpu_header))
        for (method, num_rays, particles), fields in sorted(grouped.items()):
            if "ms_gpu_range" not in fields:
                continue
            vals = fields["ms_gpu_range"]
            mean_ms = statistics.mean(vals)
            stdev_ms = statistics.stdev(vals) if len(vals) > 1 else 0.0
            print(f"{method:>8} {num_rays:>5} {particles:>10} {mean_ms:>10.3f} {stdev_ms:>10.3f}")

    print()
    print("==> Sustained particle counts at target frequencies (based on mean ms_total)")
    for (method, num_rays), points in sorted(summary.items()):
        points.sort()  # by particle count, ascending
        print(f"  -- method={method}, {num_rays} rays/particle --")
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
