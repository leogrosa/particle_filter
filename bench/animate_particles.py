#!/usr/bin/env python3
"""
Animates the particle cloud evolving over PF iterations, reading the CSVs produced by
`mcl_convergence --out-prefix DIR`. Companion to plot_convergence.py, which only shows
a handful of static snapshots -- this is the full walk-through across every logged
iteration: particle (x, y) scatter colored by weight, ground-truth pose marked each
frame, current iteration number in the title.

Coordinate note: particle/trajectory (x, y) are already in map-pixel coordinates, the
same convention the C++ benchmarks use (range_libc's OMap loads grid[x][y] straight off
the PNG's rows, so pixel y=0 is the image's top row -- matplotlib's default imshow
origin). No meters conversion needed.

Output: prefers an mp4 via the ffmpeg writer; if ffmpeg isn't available on this machine,
falls back to an animated GIF via the Pillow writer instead of crashing.

Usage:
  ./animate_particles.py --out-prefix results/convergence_run1
  ./animate_particles.py --out-prefix results/convergence_run1 --fps 5  # slow motion
"""
import argparse
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
from loguru import logger

HEADING_ARROW_LEN = 20  # px, deliberately small/subtle -- see plot_convergence.py's
                        # matching choice for the same true-pose heading arrow

MAP_PNG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "maps", "basement_fixed.png")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def find_most_recent_run(results_dir):
    """Most recently modified subdirectory of results_dir that has a timing.csv
    in it (i.e. an actual mcl_convergence --out-prefix run, not e.g. 'old/')."""
    candidates = []
    if not os.path.isdir(results_dir):
        sys.exit(f"error: results directory not found: {results_dir}")
    for name in os.listdir(results_dir):
        path = os.path.join(results_dir, name)
        if os.path.isdir(path) and os.path.exists(os.path.join(path, "timing.csv")):
            candidates.append(path)
    if not candidates:
        sys.exit(f"error: no run directories (containing timing.csv) found under {results_dir}")
    return max(candidates, key=os.path.getmtime)


def load_map():
    img = plt.imread(MAP_PNG)
    return img, img.shape[1], img.shape[0]


def require_csv(path):
    if not os.path.exists(path):
        sys.exit(f"error: expected CSV not found: {path}")
    return path


def pick_writer(requested_output):
    """Prefer the ffmpeg writer (mp4). If ffmpeg isn't on PATH, fall back to the Pillow
    writer (gif) instead of crashing, and rewrite the output extension to .gif so the
    file on disk actually matches its format."""
    root, ext = os.path.splitext(requested_output)
    if animation.writers.is_available("ffmpeg"):
        return "ffmpeg", requested_output
    print(
        "WARNING: matplotlib's ffmpeg animation writer is not available on this machine "
        "(ffmpeg not found on PATH) -- falling back to an animated GIF via the Pillow writer.",
        file=sys.stderr,
    )
    return "pillow", root + ".gif"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-prefix",
                     help="output directory used with mcl_convergence --out-prefix "
                          "(reads DIR/particles.csv, DIR/trajectory.csv)")
    ap.add_argument("--recent", "-r", action="store_true",
                     help=f"use the most recently modified run directory under {RESULTS_DIR} "
                          "instead of --out-prefix")
    ap.add_argument("--output", default=None,
                     help="output video/gif path (default: <out-prefix>/animation.mp4, "
                          "falls back to .gif if the ffmpeg writer is unavailable)")
    ap.add_argument("--fps", type=float, default=None,
                     help="playback frames per second. Default: 1/--dt, i.e. real-time "
                          "playback (one frame per iteration, each iteration is --dt "
                          "simulated seconds). Pass explicitly to play slower/faster than "
                          "real time, e.g. --fps 5 to watch convergence in slow motion.")
    ap.add_argument("--dt", type=float, default=1.0 / 40.0,
                     help="seconds per iteration, for displaying simulated elapsed time "
                          "in the title instead of a raw iteration count (default 1/40s "
                          "= 40Hz). Keep matched to whatever --dt generate_trajectory.py "
                          "or run_convergence.py used, if you want the displayed time to "
                          "mean anything physically.")
    args = ap.parse_args()
    fps = args.fps if args.fps is not None else 1.0 / args.dt

    if args.recent:
        if args.out_prefix:
            sys.exit("error: --out-prefix and --recent/-r are mutually exclusive")
        out_prefix = find_most_recent_run(RESULTS_DIR)
        print(f"using most recent run: {out_prefix}")
    elif args.out_prefix:
        out_prefix = args.out_prefix
    else:
        sys.exit("error: one of --out-prefix or --recent/-r is required")

    particles_csv = require_csv(f"{out_prefix}/particles.csv")
    trajectory_csv = require_csv(f"{out_prefix}/trajectory.csv")

    particles = pd.read_csv(particles_csv)
    traj = pd.read_csv(trajectory_csv).set_index("iter")

    requested_output = args.output or f"{out_prefix}/animation.mp4"
    writer_name, output_path = pick_writer(requested_output)

    img, w, h = load_map()
    iters_sorted = sorted(particles["iter"].unique())
    vmin, vmax = particles["weight"].min(), particles["weight"].max()

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img, extent=[0, w, h, 0])
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")

    first = particles[particles["iter"] == iters_sorted[0]]
    scat = ax.scatter(
        first["x"], first["y"], c=first["weight"], cmap="viridis",
        vmin=vmin, vmax=vmax, s=8, alpha=0.85, linewidths=0,
    )
    true_quiver = ax.quiver(
        [0], [0], [0], [0], color="red", angles="xy", scale_units="xy", scale=1,
        width=0.005, zorder=5,
    )
    true_pose_handle = Line2D(
        [0], [0], color="red", lw=1.5, marker=">", markersize=6,
        linestyle="none", label="true pose",
    )
    ax.legend(handles=[true_pose_handle], loc="upper right", fontsize=8, framealpha=0.9)
    cbar = fig.colorbar(scat, ax=ax, shrink=0.8)
    cbar.set_label("particle weight")
    title = ax.set_title("")
    fig.tight_layout()

    def update(frame_idx):
        it = iters_sorted[frame_idx]
        sub = particles[particles["iter"] == it]
        scat.set_offsets(sub[["x", "y"]].to_numpy())
        scat.set_array(sub["weight"].to_numpy())
        if it in traj.index:
            t = traj.loc[it]
            true_quiver.set_offsets([[t["x"], t["y"]]])
            true_quiver.set_UVC(
                HEADING_ARROW_LEN * math.cos(t["theta"]),
                HEADING_ARROW_LEN * math.sin(t["theta"]),
            )
        sim_t = it * args.dt
        title.set_text(
            f"t={sim_t:.3f}s  (iteration {it} / {iters_sorted[-1]}, {len(sub)} particles)"
        )
        return scat, true_quiver, title

    total_frames = len(iters_sorted)
    log_every = max(1, total_frames // 20)  # ~20 progress lines regardless of length

    def report_progress(current_frame, _total_frames):
        if current_frame % log_every == 0 or current_frame == total_frames - 1:
            pct = 100.0 * (current_frame + 1) / total_frames
            logger.info(f"rendering frame {current_frame + 1}/{total_frames} ({pct:.0f}%)")

    anim = animation.FuncAnimation(fig, update, frames=total_frames, blit=False)
    logger.info(f"rendering {total_frames} frames to {output_path} ({writer_name}, {fps:.1f}fps)")
    anim.save(output_path, writer=writer_name, fps=fps, progress_callback=report_progress)
    plt.close(fig)
    logger.info(f"wrote {output_path}")


if __name__ == "__main__":
    main()
