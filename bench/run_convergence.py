#!/usr/bin/env python3
"""
Runs the mcl_convergence binary once, for a single (particles, rays, iters, seed) configuration,
and reports where its three output CSVs landed.

This is deliberately a thin, single-shot wrapper: one configuration per invocation. It does not
sweep or compare multiple configurations -- if you want to compare particle counts, invoke this
script multiple times with different --out-prefix values and diff/plot the resulting CSVs
yourself. See bench_stats.py for that kind of multi-run driver; this script is intentionally not
that.

Binary contract (see mcl_convergence.cpp / bench/build_convergence.sh, built in parallel against
this same contract):
    mcl_convergence --particles N --rays N --iters N --seed N --out-prefix DIR [--trajectory PATH]
--out-prefix DIR is an output directory (created if missing, by both the binary itself and this
script) holding DIR/timing.csv, DIR/particles.csv, DIR/trajectory.csv. If --trajectory is omitted,
the binary auto-generates its own stand-still ground-truth trajectory (one free-space pose sampled
from its own map + --seed) and still writes DIR/trajectory.csv -- this script never needs to
generate or touch a trajectory file itself. --trajectory is plumbed through here purely so a
future phase can pass a real trajectory file in via a config change, not a code change.

The binary runs via Popen, not a blocking call: its stdout is streamed live and re-logged as it
arrives, and DIR/timing.csv is polled on a timer to report the first iteration's timing and
roughly every iters//20 iterations after that -- both best-effort, since no flush was added to
the binary's own CSV writes, so rows may arrive in bursts rather than smoothly.
"""
import argparse
import math
import subprocess
import sys
import threading
import time
from datetime import datetime
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
            iter_val, _, _, _, _, ms_range_sensor, ess = rows[n - 1].split(",")
            logger.info(f"    iter {iter_val}/{iters} ({100 * n // iters}%) -- ms_range_sensor={float(ms_range_sensor):.3f} ess={float(ess):.1f}")
            last_reported = n
            while next_threshold <= n:
                next_threshold += progress_interval


def run_binary(particles, rays, iters, seed, out_prefix, trajectory, dt, squash_factor,
               roughening_k, ess_resampling_threshold, init_mode, init_std_xy, init_std_theta,
               disable_measurement_update, motion_dispersion_x, motion_dispersion_y,
               motion_dispersion_theta):
    cmd = [
        str(BINARY),
        "--particles", str(particles),
        "--rays", str(rays),
        "--iters", str(iters),
        "--seed", str(seed),
        "--out-prefix", str(out_prefix),
        "--dt", str(dt),
        "--squash-factor", str(squash_factor),
        "--roughening-k", str(roughening_k),
        "--ess-resampling-threshold", str(ess_resampling_threshold),
        "--init-mode", str(init_mode),
        "--init-std-xy", str(init_std_xy),
        "--init-std-theta", str(init_std_theta),
        "--disable-measurement-update", "1" if disable_measurement_update else "0",
        "--motion-dispersion-x", str(motion_dispersion_x),
        "--motion-dispersion-y", str(motion_dispersion_y),
        "--motion-dispersion-theta", str(motion_dispersion_theta),
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

    timing_path = Path(out_prefix) / "timing.csv"
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
    ap.add_argument("--iters", type=int, default=None,
                     help="number of MCL iterations (default 60, unless --time is given). "
                          "Mutually exclusive with --time.")
    ap.add_argument("--time", type=float, default=None,
                     help="simulation duration in seconds, as an alternative to --iters -- "
                          "this script computes iters = ceil(time / dt). Mutually exclusive "
                          "with --iters.")
    ap.add_argument("--dt", type=float, default=1.0 / 40.0,
                     help="seconds per iteration (default 1/40s = 40Hz). Used here to convert "
                          "--time into an iteration count, AND forwarded to the binary's own "
                          "--dt, which it uses to convert each particle's per-iteration odometry "
                          "speed reading into a pixel displacement. If you're also passing "
                          "--trajectory from generate_trajectory.py, keep this matched to "
                          "that script's --dt, or 'N seconds of simulation' here won't "
                          "actually correspond to N seconds of the ground-truth robot's "
                          "motion (mcl_convergence.cpp indexes the trajectory CSV by row "
                          "number, not by its t column).")
    ap.add_argument("--squash-factor", type=float, default=2.2,
                     help="squash the raw measurement-update weight by raising it to "
                          "1/squash_factor before normalizing, forwarded to the binary's own "
                          "--squash-factor (see mcl_convergence.cpp's squash_weights(), "
                          "docs/Lab5.pdf sec 3.2). Default 2.2 matches MIT particle_filter.py's "
                          "own default (launch/localize.launch); squash_factor=1 disables "
                          "squashing.")
    ap.add_argument("--roughening-k", type=float, default=0.2,
                     help="roughening tuning constant K (Gordon, Salmond & Smith 1993), "
                          "forwarded to the binary's own --roughening-k. "
                          "sigma_i = K * E_i * N^(-1/d) added to each particle dimension "
                          "right after resampling. Default 0.2; K=0 disables roughening.")
    ap.add_argument("--ess-resampling-threshold", type=float, default=0.5,
                     help="skip resampling (and roughening) when ESS is above this fraction "
                          "of N, forwarded to the binary's own --ess-resampling-threshold "
                          "(see mcl_convergence.cpp's resample skip logic, "
                          "papers/16831_lecture05_gseyfarth_zbatts.pdf). Default 0.2; 0 "
                          "disables skipping (always resample).")
    ap.add_argument("--init-mode", choices=["global", "tracking"], default="global",
                     help="forwarded to the binary's own --init-mode. 'global' (default) "
                          "scatters the initial population uniformly across all free-space "
                          "cells (global localization). 'tracking' draws it from a Gaussian "
                          "centered on the ground-truth pose at iteration 0 (--init-std-xy / "
                          "--init-std-theta), simulating a known starting pose.")
    ap.add_argument("--init-std-xy", type=float, default=0.5,
                     help="position std dev in meters for --init-mode tracking, forwarded to "
                          "the binary's own --init-std-xy. Default 0.5.")
    ap.add_argument("--init-std-theta", type=float, default=0.4,
                     help="heading std dev in radians for --init-mode tracking, forwarded to "
                          "the binary's own --init-std-theta. Default 0.4.")
    ap.add_argument("--disable-measurement-update", action="store_true",
                     help="skip measurement_update/squash/normalize entirely and set weights "
                          "uniform every iteration (ESS reads N, so resample/roughen self-skip "
                          "too), forwarded to the binary's own --disable-measurement-update -- "
                          "isolates the motion model alone, no sensor feedback, for debugging.")
    ap.add_argument("--motion-dispersion-x", type=float, default=0.05,
                     help="residual isotropic x noise std dev in meters, forwarded to the "
                          "binary's own --motion-dispersion-x (converted to pixels internally "
                          "via MAP_RESOLUTION). Default 0.05, matching MIT's own "
                          "motion_dispersion_x ROS param. Pass 0 for a zero-process-noise "
                          "isolation run.")
    ap.add_argument("--motion-dispersion-y", type=float, default=0.025,
                     help="same as --motion-dispersion-x, y axis. Default 0.025.")
    ap.add_argument("--motion-dispersion-theta", type=float, default=0.25,
                     help="residual heading noise std dev in radians (added on top of "
                          "true_delta_theta each iteration), forwarded to the binary's own "
                          "--motion-dispersion-theta. Default 0.25, matching MIT's own "
                          "motion_dispersion_theta ROS param.")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed (default 42)")
    ap.add_argument("--out-prefix", default=None,
                     help="output directory; the binary writes <dir>/timing.csv, "
                          "<dir>/particles.csv, <dir>/trajectory.csv (created if missing). "
                          "Default: bench/results/<YYYY-MM-DD_HHMMSS>")
    ap.add_argument("--trajectory", default=None,
                     help="optional path to a ground-truth trajectory CSV, forwarded to the binary "
                          "as --trajectory. Unused today (the binary auto-generates a stand-still "
                          "trajectory when omitted); plumbed through for a future phase.")
    args = ap.parse_args()

    if args.time is not None and args.iters is not None:
        ap.error("--time and --iters are mutually exclusive -- pass one or the other")
    if args.time is not None:
        iters = math.ceil(args.time / args.dt)
        logger.info(f"==> --time {args.time}s @ dt={args.dt}s -> {iters} iterations")
    elif args.iters is not None:
        iters = args.iters
    else:
        iters = 60

    if args.out_prefix is None:
        out_prefix = DEFAULT_RESULTS_DIR / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    else:
        out_prefix = Path(args.out_prefix)
    out_prefix.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"==> Config: particles={args.particles} rays={args.rays} iters={iters} "
        f"seed={args.seed} squash_factor={args.squash_factor} roughening_k={args.roughening_k} "
        f"ess_resampling_threshold={args.ess_resampling_threshold} init_mode={args.init_mode} "
        f"disable_measurement_update={args.disable_measurement_update} "
        f"motion_dispersion_x={args.motion_dispersion_x} motion_dispersion_y={args.motion_dispersion_y} "
        f"motion_dispersion_theta={args.motion_dispersion_theta} out_prefix={out_prefix}"
    )
    if args.init_mode == "tracking":
        logger.info(
            f"==> init_std_xy={args.init_std_xy}m init_std_theta={args.init_std_theta}rad"
        )
    if args.trajectory is not None:
        logger.info(f"==> Forwarding --trajectory {args.trajectory}")

    build_binary()
    run_binary(args.particles, args.rays, iters, args.seed, out_prefix, args.trajectory, args.dt,
               args.squash_factor, args.roughening_k, args.ess_resampling_threshold,
               args.init_mode, args.init_std_xy, args.init_std_theta,
               args.disable_measurement_update, args.motion_dispersion_x,
               args.motion_dispersion_y, args.motion_dispersion_theta)

    timing_csv = out_prefix / "timing.csv"
    particles_csv = out_prefix / "particles.csv"
    trajectory_csv = out_prefix / "trajectory.csv"
    logger.info("==> Done. Output CSVs:")
    logger.info(f"    timing:     {timing_csv}")
    logger.info(f"    particles:  {particles_csv}")
    logger.info(f"    trajectory: {trajectory_csv}")


if __name__ == "__main__":
    main()
