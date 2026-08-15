#!/usr/bin/env python3
"""
Renders two static figures from the CSVs produced by `mcl_convergence --out-prefix DIR`:

  <base>/convergence.png -- 3 stacked subplots (distinct_cells, mean_dist_to_true w/
                             stddev_x/stddev_y, ms_range_sensor) vs iteration number,
                             read from DIR/timing.csv only.
  <base>/snapshots.png   -- small-multiples scatter of the particle cloud at a handful
                             of iterations (first / ~25% / ~50% / ~75% / last, adapted
                             to however many iterations are actually present), colored
                             by weight, with the ground-truth pose marked. Read from
                             DIR/particles.csv + DIR/trajectory.csv.

`<base>` is --output if given, else --out-prefix -- so by default this produces exactly
<out-prefix>/convergence.png and <out-prefix>/snapshots.png, alongside the CSVs.

Coordinate note: particle/trajectory (x, y) are already in map-pixel coordinates, the
same convention the C++ benchmarks use (range_libc's OMap loads grid[x][y] straight off
the PNG's rows, so pixel y=0 is the image's top row -- matplotlib's default imshow
origin). No meters conversion needed; we just imshow the map and scatter (x, y) on top
with a matching extent.

Usage:
  ./plot_convergence.py --out-prefix results/convergence_run1
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

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
    h, w = img.shape[0], img.shape[1]
    return img, w, h


def require_csv(path):
    if not os.path.exists(path):
        sys.exit(f"error: expected CSV not found: {path}")
    return path


def plot_convergence_panel(timing_csv, output, tag):
    df = pd.read_csv(timing_csv)

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

    ax = axes[0]
    ax.plot(df["iter"], df["distinct_cells"], color="#1f77b4", marker="o", markersize=3)
    ax.set_ylabel("distinct cells touched")
    ax.set_title("Working-set size: distinct map cells touched by lookups this iteration")
    ax.grid(alpha=0.3)

    # mean_dist_to_true and stddev_x/stddev_y are all pixel-distance quantities, so they
    # share one axis honestly (no need for -- and no use of -- a second y-scale). We plot
    # stddev_x/y as separate lines rather than a +/- band around mean_dist_to_true: the
    # latter is a mean *distance*, not a mean position, so a band built from stddev_x/y
    # would visually imply a stat we don't actually have.
    ax = axes[1]
    ax.plot(df["iter"], df["mean_dist_to_true"], color="#d62728", lw=2, label="mean dist to true pose")
    ax.plot(df["iter"], df["stddev_x"], color="#2ca02c", lw=1.25, ls="--", label="stddev_x")
    ax.plot(df["iter"], df["stddev_y"], color="#9467bd", lw=1.25, ls="--", label="stddev_y")
    ax.set_ylabel("pixels")
    ax.set_title("Convergence: population spread around the true pose")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(df["iter"], df["ms_range_sensor"], color="#ff7f0e", marker="o", markersize=3)
    ax.set_xlabel("iteration")
    ax.set_ylabel("ms_range_sensor\n(local dev-machine, directional only)")
    ax.set_title("Range-sensor lookup wall-clock -- NOT a hardware-calibrated figure")
    ax.grid(alpha=0.3)

    fig.suptitle(f"PF convergence vs. lookup working set -- {tag}", fontsize=13)
    fig.text(
        0.5, 0.005,
        "Note: ms_range_sensor is a local Mac timing meant for directional comparison "
        "only -- it is not calibrated to any target hardware.",
        ha="center", fontsize=8, style="italic",
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def pick_snapshot_iters(iters_sorted, n=5):
    """First / ~25% / ~50% / ~75% / last, deduplicated and adapted to len(iters_sorted)."""
    n = min(n, len(iters_sorted))
    if n <= 1:
        return iters_sorted[:1]
    idxs = [round(i * (len(iters_sorted) - 1) / (n - 1)) for i in range(n)]
    seen = set()
    picked = []
    for idx in idxs:
        it = iters_sorted[idx]
        if it not in seen:
            seen.add(it)
            picked.append(it)
    return picked


def plot_snapshots_panel(particles_csv, trajectory_csv, output, tag, n_snapshots=5):
    particles = pd.read_csv(particles_csv)
    traj = pd.read_csv(trajectory_csv).set_index("iter")
    img, w, h = load_map()

    iters_sorted = sorted(particles["iter"].unique())
    snap_iters = pick_snapshot_iters(iters_sorted, n_snapshots)
    ncols = len(snap_iters)

    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4.6), squeeze=False)
    axes = axes[0]

    vmin, vmax = particles["weight"].min(), particles["weight"].max()
    sc = None
    for ax, it in zip(axes, snap_iters):
        ax.imshow(img, extent=[0, w, h, 0])
        sub = particles[particles["iter"] == it]
        sc = ax.scatter(
            sub["x"], sub["y"], c=sub["weight"], cmap="viridis",
            vmin=vmin, vmax=vmax, s=6, alpha=0.85, linewidths=0,
        )
        if it in traj.index:
            t = traj.loc[it]
            ax.scatter(
                [t["x"]], [t["y"]], marker="*", s=200, c="red",
                edgecolors="black", linewidths=0.8, label="true pose", zorder=5,
            )
        ax.set_title(f"iter {it}")
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.set_xticks([])
        ax.set_yticks([])

    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=1, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    if sc is not None:
        cbar = fig.colorbar(sc, ax=axes.tolist(), shrink=0.75, pad=0.015)
        cbar.set_label("particle weight")

    fig.suptitle(f"Particle cloud snapshots -- {tag}", fontsize=13)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-prefix",
                     help="output directory used with mcl_convergence --out-prefix "
                          "(reads DIR/timing.csv, DIR/particles.csv, DIR/trajectory.csv)")
    ap.add_argument("--recent", "-r", action="store_true",
                     help=f"use the most recently modified run directory under {RESULTS_DIR} "
                          "instead of --out-prefix")
    ap.add_argument("--output", default=None,
                     help="base output path; produces <output>/convergence.png and "
                          "<output>/snapshots.png (default: same as --out-prefix)")
    args = ap.parse_args()

    if args.recent:
        if args.out_prefix:
            sys.exit("error: --out-prefix and --recent/-r are mutually exclusive")
        out_prefix = find_most_recent_run(RESULTS_DIR)
        print(f"using most recent run: {out_prefix}")
    elif args.out_prefix:
        out_prefix = args.out_prefix
    else:
        sys.exit("error: one of --out-prefix or --recent/-r is required")

    timing_csv = require_csv(f"{out_prefix}/timing.csv")
    particles_csv = require_csv(f"{out_prefix}/particles.csv")
    trajectory_csv = require_csv(f"{out_prefix}/trajectory.csv")

    base = args.output or out_prefix
    tag = os.path.basename(os.path.normpath(out_prefix))

    convergence_out = plot_convergence_panel(timing_csv, f"{base}/convergence.png", tag)
    print(f"wrote {convergence_out}")

    snapshots_out = plot_snapshots_panel(particles_csv, trajectory_csv, f"{base}/snapshots.png", tag)
    print(f"wrote {snapshots_out}")


if __name__ == "__main__":
    main()
