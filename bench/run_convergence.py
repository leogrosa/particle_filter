#!/usr/bin/env python3
"""
Runs the mcl_convergence binary once, for a single (particles, rays, iters, seed) configuration,
and reports where its three output CSVs landed.

This is deliberately a thin, single-shot wrapper: one configuration per invocation, run serially
(build, then one blocking subprocess call, then done). It does not sweep or compare multiple
configurations -- if you want to compare particle counts, invoke this script multiple times with
different --out-prefix values and diff/plot the resulting CSVs yourself. See bench_stats.py for
that kind of multi-run driver; this script is intentionally not that.

Binary contract (see mcl_convergence.cpp / bench/build_convergence.sh, built in parallel against
this same contract):
    mcl_convergence --particles N --rays N --iters N --seed N --out-prefix PATH [--trajectory PATH]
Given --out-prefix PATH, it writes PATH_timing.csv, PATH_particles.csv, PATH_trajectory.csv. If
--trajectory is omitted, the binary auto-generates its own stand-still ground-truth trajectory
(one free-space pose sampled from its own map + --seed) and still writes PATH_trajectory.csv --
this script never needs to generate or touch a trajectory file itself. --trajectory is plumbed
through here purely so a future phase can pass a real trajectory file in via a config change, not
a code change.
"""
import argparse
import subprocess
import sys
import threading
import time
from pathlib import Path

from loguru import logger

BENCH_DIR = Path(__file__).resolve().parent
BUILD_SCRIPT = BENCH_DIR / "build_convergence.sh"
BINARY = BENCH_DIR / "build_convergence" / "bin" / "mcl_convergence"
DEFAULT_RESULTS_DIR = BENCH_DIR / "results"


def build_binary():
    logger.info(f"==> Ensuring {BINARY} is built (running {BUILD_SCRIPT})")
    try:
        result = subprocess.run(
            [str(BUILD_SCRIPT)],
            cwd=BENCH_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError:
        logger.error(f"build script not found at {BUILD_SCRIPT}")
        sys.exit(1)
    if result.stdout.strip():
        logger.info(result.stdout.rstrip())
    if result.returncode != 0:
        logger.error(f"build_convergence.sh failed with exit code {result.returncode}")
        if result.stderr.strip():
            logger.error(result.stderr.rstrip())
        sys.exit(1)
    if result.stderr.strip():
        # Build succeeded but produced stderr output (e.g. compiler warnings) -- surface it
        # without treating it as a hard failure.
        logger.warning(result.stderr.rstrip())
    logger.info("==> Build ok")


def _stream_stdout(proc):
    """Reads the binary's stdout line by line as it arrives and re-logs each line via loguru.
    Runs in its own thread since a plain 'for line in proc.stdout' would block until the
    binary's own stdio buffer flushes -- interleaving this with the timing.csv poll loop below
    needs the two to run concurrently, not one after another."""
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            logger.info(f"[mcl_convergence] {line}")


def _poll_timing_progress(timing_path, iters, stop_event):
    """Polls timing_path's row count while the binary runs, logging the first iteration's row
    as soon as it appears and then roughly every iters//20 rows after that. No fflush was added
    on the C++ side, so rows only become visible here whenever the binary's own stdio buffer
    happens to flush -- expect this to arrive in bursts, not smoothly, especially for the first
    stretch of iterations."""
    progress_interval = max(1, iters // 20)
    next_threshold = 1
    last_reported = 0
    while not stop_event.is_set():
        time.sleep(1.0)
        if not timing_path.exists():
            continue
        with open(timing_path) as f:
            rows = f.read().splitlines()[1:]  # skip header
        n = len(rows)
        if n > last_reported and n >= next_threshold:
            iter_val, _, _, _, _, ms_range_sensor = rows[n - 1].split(",")
            logger.info(f"    iter {iter_val}/{iters} ({100 * n // iters}%) -- ms_range_sensor={float(ms_range_sensor):.3f}")
            last_reported = n
            while next_threshold <= n:
                next_threshold += progress_interval


def run_binary(particles, rays, iters, seed, out_prefix, trajectory):
    cmd = [
        str(BINARY),
        "--particles", str(particles),
        "--rays", str(rays),
        "--iters", str(iters),
        "--seed", str(seed),
        "--out-prefix", str(out_prefix),
    ]
    if trajectory is not None:
        cmd += ["--trajectory", str(trajectory)]

    logger.info(f"==> Running: {' '.join(cmd)}")
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except FileNotFoundError:
        logger.error(f"binary not found at {BINARY} -- build_convergence.sh should have produced it")
        sys.exit(1)

    timing_path = Path(f"{out_prefix}_timing.csv")
    stop_event = threading.Event()
    stdout_thread = threading.Thread(target=_stream_stdout, args=(proc,), daemon=True)
    progress_thread = threading.Thread(target=_poll_timing_progress, args=(timing_path, iters, stop_event), daemon=True)
    stdout_thread.start()
    progress_thread.start()

    returncode = proc.wait()
    stop_event.set()
    stdout_thread.join()
    progress_thread.join()

    if returncode != 0:
        logger.error(f"mcl_convergence exited with code {returncode}")
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--particles", type=int, default=2000, help="number of particles (default 2000)")
    ap.add_argument("--rays", type=int, default=60, help="number of LIDAR rays (default 60)")
    ap.add_argument("--iters", type=int, default=60, help="number of MCL iterations (default 60)")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed (default 42)")
    ap.add_argument("--out-prefix", default=None,
                     help="output path prefix; the binary writes <prefix>_timing.csv, "
                          "<prefix>_particles.csv, <prefix>_trajectory.csv. Default: "
                          "bench/results/convergence_p{particles}_r{rays}_i{iters}_s{seed}")
    ap.add_argument("--trajectory", default=None,
                     help="optional path to a ground-truth trajectory CSV, forwarded to the binary "
                          "as --trajectory. Unused today (the binary auto-generates a stand-still "
                          "trajectory when omitted); plumbed through for a future phase.")
    args = ap.parse_args()

    if args.out_prefix is None:
        out_prefix = DEFAULT_RESULTS_DIR / (
            f"convergence_p{args.particles}_r{args.rays}_i{args.iters}_s{args.seed}"
        )
    else:
        out_prefix = Path(args.out_prefix)

    logger.info(
        f"==> Config: particles={args.particles} rays={args.rays} iters={args.iters} "
        f"seed={args.seed} out_prefix={out_prefix}"
    )
    if args.trajectory is not None:
        logger.info(f"==> Forwarding --trajectory {args.trajectory}")

    build_binary()
    run_binary(args.particles, args.rays, args.iters, args.seed, out_prefix, args.trajectory)

    timing_csv = Path(f"{out_prefix}_timing.csv")
    particles_csv = Path(f"{out_prefix}_particles.csv")
    trajectory_csv = Path(f"{out_prefix}_trajectory.csv")
    logger.info("==> Done. Output CSVs:")
    logger.info(f"    timing:     {timing_csv}")
    logger.info(f"    particles:  {particles_csv}")
    logger.info(f"    trajectory: {trajectory_csv}")


if __name__ == "__main__":
    main()
