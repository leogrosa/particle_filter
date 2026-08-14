#!/usr/bin/env python3
"""
Animates the particle cloud evolving over PF iterations, reading the CSVs produced by
`mcl_convergence --out-prefix PATH`. Companion to plot_convergence.py, which only shows
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
  ./animate_particles.py --out-prefix results/convergence_run1 --fps 5
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import pandas as pd

MAP_PNG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "maps", "basement_fixed.png")


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
    ap.add_argument("--out-prefix", required=True,
                     help="shared prefix used with mcl_convergence --out-prefix "
                          "(reads PATH_particles.csv, PATH_trajectory.csv)")
    ap.add_argument("--output", default=None,
                     help="output video/gif path (default: <out-prefix>_animation.mp4, "
                          "falls back to .gif if the ffmpeg writer is unavailable)")
    ap.add_argument("--fps", type=int, default=5,
                     help="playback frames per second (default 5 -- this is for watching "
                          "convergence happen, not real-time video)")
    args = ap.parse_args()

    particles_csv = require_csv(f"{args.out_prefix}_particles.csv")
    trajectory_csv = require_csv(f"{args.out_prefix}_trajectory.csv")

    particles = pd.read_csv(particles_csv)
    traj = pd.read_csv(trajectory_csv).set_index("iter")

    requested_output = args.output or f"{args.out_prefix}_animation.mp4"
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
    true_marker, = ax.plot(
        [], [], marker="*", markersize=16, color="red",
        markeredgecolor="black", linestyle="none", label="true pose",
    )
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
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
            true_marker.set_data([t["x"]], [t["y"]])
        title.set_text(f"iteration {it} / {iters_sorted[-1]}  ({len(sub)} particles)")
        return scat, true_marker, title

    anim = animation.FuncAnimation(fig, update, frames=len(iters_sorted), blit=False)
    anim.save(output_path, writer=writer_name, fps=args.fps)
    plt.close(fig)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
