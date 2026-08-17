#!/usr/bin/env python3
"""
Turns a hand-drawn pink/magenta circuit on a map PNG into a ground-truth
trajectory CSV consumable by mcl_convergence.cpp's --trajectory flag
(iter,t,x,y,theta,vx,vy).

Pipeline:
  1. Threshold the map image for pink/magenta pixels -> a thick-stroke mask.
  2. Skeletonize the mask down to a 1px-wide centerline.
  3. Order the skeleton pixels into a single closed loop via greedy
     nearest-neighbor chaining (works because the drawn loop doesn't
     self-intersect -- a self-crossing path would need a real graph-based
     ordering instead, not attempted here).
  4. Arc-length parameterize the ordered loop in real-world meters, using the
     map's documented resolution (basement_fixed.map.yaml: 0.0504 m/px --
     confirmed against RangeLibc's own MAP_RESOLUTION constant and the
     CDDT-paper-Fig-12 P1->P2 velocity sanity check, 2026-08-15).
  5. Walk the loop at constant speed --velocity, sampling every --dt seconds,
     for --iters samples, wrapping around the loop for as many laps as
     needed.
  6. theta at each sample = direction of travel (atan2(dy, dx) between
     consecutive samples), matching RangeLib.h's own convention
     (ray_direction_x=cosf(heading), ray_direction_y=sinf(heading), pixel y
     grows downward -- confirmed by reading RangeLib.h:937-938 directly, not
     assumed). Iteration 0 has no previous sample, so uses a fixed initial
     heading: 3*pi/2, i.e. "up" on screen / north (theta=0 is +x/east,
     theta=pi/2 is +y which is *south* since pixel y grows downward, so north
     is theta=-pi/2 == 3*pi/2 in [0, 2*pi) form).

Usage:
  ./generate_trajectory.py --map maps/basement_trajectory.png --iters 400 \
      --out maps/basement_loop_trajectory.csv
"""
import argparse
import math
import sys

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

MAP_RESOLUTION = 0.0504  # m/px, from basement_fixed.map.yaml -- see module docstring
NORTH_THETA = 3.0 * math.pi / 2.0  # "up" on screen, in RangeLib's theta convention

# Measured directly off the drawn stroke in basement_trajectory.png (2026-08-15):
# pure-ish magenta, sampled pixels came back (251, 2, 255). Generous tolerance
# to survive anti-aliasing at the stroke's edges.
PINK_RGB = (251, 2, 255)
PINK_TOL = 60


def extract_pink_mask(img):
    r, g, b = img[..., 0].astype(int), img[..., 1].astype(int), img[..., 2].astype(int)
    pr, pg, pb = PINK_RGB
    mask = (
        (np.abs(r - pr) < PINK_TOL)
        & (np.abs(g - pg) < PINK_TOL)
        & (np.abs(b - pb) < PINK_TOL)
    )
    return mask


def order_skeleton_points(ys, xs, max_step_px=3.0):
    """Greedy nearest-neighbor chaining into a single path. Rebuilds the
    KDTree periodically (every rebuild_every steps) since points are removed
    as they're visited -- querying a stale tree just risks re-finding
    already-visited points, not correctness, but rebuilding keeps queries
    cheap as the remaining set shrinks."""
    points = np.stack([xs, ys], axis=1).astype(float)
    remaining = list(range(len(points)))
    order = [remaining.pop(0)]
    rebuild_every = 200
    tree = None
    since_rebuild = 0
    while remaining:
        if tree is None or since_rebuild >= rebuild_every:
            tree = cKDTree(points[remaining])
            since_rebuild = 0
        cur = points[order[-1]]
        dist, idx = tree.query(cur, k=1)
        since_rebuild += 1
        if dist > max_step_px:
            # Nearest remaining point is too far to be a continuation of this
            # stroke -- likely a disconnected artifact (stray anti-aliased
            # pixel, small skeletonize spur). Stop; the loop is done.
            break
        chosen = remaining.pop(idx)
        order.append(chosen)
        tree = None  # remaining changed; force rebuild next iteration
    return points[order]


def arc_length_cumulative(loop_px, resolution):
    deltas = np.diff(loop_px, axis=0, append=loop_px[:1])
    seg_len_m = np.linalg.norm(deltas, axis=1) * resolution
    cum = np.concatenate([[0.0], np.cumsum(seg_len_m)])
    return cum  # cum[-1] is the full loop length in meters (cum[i] is arc length up to point i)


def sample_trajectory(loop_px, resolution, velocity, dt, iters):
    cum = arc_length_cumulative(loop_px, resolution)
    total_length_m = cum[-1]
    if total_length_m <= 0:
        sys.exit("error: degenerate loop (zero arc length) -- check the extracted path")

    poses = []
    prev_xy = None
    for i in range(iters):
        target_m = (velocity * dt * i) % total_length_m
        seg_idx = int(np.searchsorted(cum, target_m, side="right") - 1)
        seg_idx = min(seg_idx, len(loop_px) - 2)
        seg_start_m, seg_end_m = cum[seg_idx], cum[seg_idx + 1]
        frac = 0.0 if seg_end_m == seg_start_m else (target_m - seg_start_m) / (seg_end_m - seg_start_m)
        xy_px = loop_px[seg_idx] + frac * (loop_px[seg_idx + 1] - loop_px[seg_idx])

        if prev_xy is None:
            theta = NORTH_THETA
        else:
            dx, dy = xy_px - prev_xy
            theta = math.atan2(dy, dx) if (dx or dy) else poses[-1][2]
        poses.append((xy_px[0], xy_px[1], theta))
        prev_xy = xy_px

    return poses


def write_trajectory_csv(path, poses, velocity, dt):
    with open(path, "w") as f:
        f.write("iter,t,x,y,theta,vx,vy\n")
        for i, (x, y, theta) in enumerate(poses):
            vx, vy = velocity * math.cos(theta), velocity * math.sin(theta)
            f.write(f"{i},{i * dt:.6f},{x:.6f},{y:.6f},{theta:.6f},{vx:.6f},{vy:.6f}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", default="maps/basement_trajectory.png",
                     help="map PNG with the pink/magenta circuit drawn on it")
    ap.add_argument("--out", required=True, help="output trajectory CSV path")
    ap.add_argument("--velocity", type=float, default=1.8,
                     help="constant speed in m/s (default 1.8, from the CDDT-paper Fig.12 "
                          "P1->P2 sanity check, 2026-08-15)")
    ap.add_argument("--dt", type=float, default=1.0 / 40.0,
                     help="seconds per iteration (default 1/40s, matching the project's "
                          "nominal 40Hz real-time MCL budget)")
    ap.add_argument("--iters", type=int, required=True,
                     help="number of trajectory rows to generate -- pass the same value "
                          "you'll use for mcl_convergence's --iters")
    ap.add_argument("--resolution", type=float, default=MAP_RESOLUTION,
                     help=f"map resolution in m/px (default {MAP_RESOLUTION})")
    ap.add_argument("--debug-plot", default=None,
                     help="optional path to save a PNG overlaying the ordered/resampled "
                          "path on the map, for visual sanity-checking")
    args = ap.parse_args()

    img = np.array(Image.open(args.map).convert("RGB"))
    mask = extract_pink_mask(img)
    n_pink = int(mask.sum())
    print(f"[1/5] pink mask: {n_pink} px")
    if n_pink == 0:
        sys.exit("error: no pink pixels found -- check PINK_RGB/PINK_TOL against the actual "
                  "drawn color")

    skeleton = skeletonize(mask)
    ys, xs = np.where(skeleton)
    print(f"[2/5] skeletonized: {len(xs)} px")

    loop_px = order_skeleton_points(ys, xs)
    print(f"[3/5] ordered into a path: {len(loop_px)} px "
          f"({'closed loop' if len(loop_px) > 0.9 * len(xs) else 'WARNING: stopped early, check for a break/spur'})")

    cum = arc_length_cumulative(loop_px, args.resolution)
    print(f"[4/5] loop length: {cum[-1]:.2f} m")

    poses = sample_trajectory(loop_px, args.resolution, args.velocity, args.dt, args.iters)
    laps = (args.velocity * args.dt * args.iters) / cum[-1]
    print(f"[5/5] sampled {len(poses)} poses (dt={args.dt:.4f}s, v={args.velocity}m/s, "
          f"~{laps:.2f} laps total)")

    write_trajectory_csv(args.out, poses, args.velocity, args.dt)
    print(f"wrote {args.out}")

    if args.debug_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 9))
        ax.imshow(img)
        xs_p = [p[0] for p in poses]
        ys_p = [p[1] for p in poses]
        ax.plot(loop_px[:, 0], loop_px[:, 1], color="cyan", lw=0.5, alpha=0.5, label="ordered skeleton")
        ax.scatter(xs_p, ys_p, c=np.arange(len(poses)), cmap="viridis", s=4, label="sampled poses")
        ax.scatter([xs_p[0]], [ys_p[0]], c="red", marker="*", s=150, zorder=5, label="iter 0")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title(f"Sampled trajectory ({len(poses)} iters, {laps:.2f} laps)")
        fig.savefig(args.debug_plot, dpi=150, bbox_inches="tight")
        print(f"wrote {args.debug_plot}")


if __name__ == "__main__":
    main()
