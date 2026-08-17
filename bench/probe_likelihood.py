#!/usr/bin/env python3
"""
Runs the likelihood_probe binary for ONE probe pose against a synthetic ground-truth
observation, and prints the resulting per-ray CSV as a colored table in the terminal.

Deliberately minimal, single-pose: no built-in two-pose comparison. To compare a true
pose against a suspect wrong pose, run this script twice (same --gt-x/--gt-y/--gt-theta
both times, different --x/--y/--theta) and eyeball the two tables / final cum_log_prob.

Binary contract (see likelihood_probe.cpp):
    likelihood_probe --gt-x X --gt-y Y --gt-theta T --x X --y Y --theta T [--rays N]
stdout: CSV header "ray,angle,obs_r,pred_d,prob,category,cum_log_prob", one row per ray.
stderr: [setup] progress lines, not part of the CSV.

category is which beam-model term (hit/short/max/rand) numerically dominates that ray's
probability -- "max" means the sensor model is explaining this ray as a no-return/open-
space reading, not a matched wall.

Pass --plot to also open a matplotlib window with both poses (position + heading arrow)
drawn on the map -- position only, not the ray fan itself.
"""
import argparse
import math
import subprocess
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

BENCH_DIR = Path(__file__).resolve().parent
BUILD_DIR = BENCH_DIR / "build_convergence"
BINARY = BUILD_DIR / "bin" / "likelihood_probe"

# The SAME map likelihood_probe.cpp actually ray-casts against (CMakeLists.txt's
# MAP_PATH) -- deliberately NOT animate_particles.py's basement_fixed.png. That
# variant recolors the "unmapped/uncertain" gray region dark enough to read as
# occupied in the ray-casting map, so a spot that looks like open corridor in
# basement_fixed.png can still be a wall here -- confirmed 2026-08-18 as the
# cause of every ray reading ~0 for an apparently-free pose. Show the map the
# binary actually sees, not a prettier stand-in.
MAP_PNG = BENCH_DIR / ".." / "maps" / "basement_fixed_unmapped120.png"
HEADING_ARROW_LEN = 30  # px

# ANSI colors, one per beam-model category -- kept to plain escape codes (no extra
# dependency) since this project's other scripts don't already depend on a terminal-
# color library.
CATEGORY_COLOR = {
    "hit": "\033[32m",    # green: matched the predicted wall
    "short": "\033[33m",  # yellow: unexpected closer obstacle
    "max": "\033[36m",    # cyan: no-return / open-space reading ("miss")
    "rand": "\033[35m",   # magenta: unexplained / noise floor
}
RESET = "\033[0m"
BOLD = "\033[1m"


def build_binary():
    logger.info(f"==> Ensuring {BINARY} is built")
    try:
        subprocess.run(
            ["cmake", "--build", str(BUILD_DIR), "--target", "likelihood_probe"],
            check=True,
        )
    except FileNotFoundError:
        logger.error(f"build directory not found at {BUILD_DIR} -- run build_convergence.sh first")
        sys.exit(1)
    except subprocess.CalledProcessError:
        logger.error("build failed")
        sys.exit(1)


def run_probe(gt_x, gt_y, gt_theta, x, y, theta, rays):
    cmd = [
        str(BINARY),
        "--gt-x", str(gt_x), "--gt-y", str(gt_y), "--gt-theta", str(gt_theta),
        "--x", str(x), "--y", str(y), "--theta", str(theta),
        "--rays", str(rays),
    ]
    logger.info(f"==> Running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        logger.error(f"likelihood_probe exited with code {proc.returncode}")
        sys.stderr.write(proc.stderr)
        sys.exit(1)
    for line in proc.stderr.splitlines():
        logger.info(f"  [probe] {line}")
    return proc.stdout


def print_table(csv_text):
    from io import StringIO
    df = pd.read_csv(StringIO(csv_text))

    header = f"{'ray':>4} {'angle(deg)':>11} {'obs_r':>6} {'pred_d':>7} {'prob':>12} {'category':>8} {'cum_log_prob':>13}"
    print(BOLD + header + RESET)
    print("-" * len(header))
    for _, row in df.iterrows():
        color = CATEGORY_COLOR.get(row["category"], "")
        line = (
            f"{int(row['ray']):>4} {row['angle'] * 180.0 / 3.14159265:>11.2f} "
            f"{int(row['obs_r']):>6} {int(row['pred_d']):>7} {row['prob']:>12.4e} "
            f"{row['category']:>8} {row['cum_log_prob']:>13.3f}"
        )
        print(color + line + RESET)

    counts = df["category"].value_counts()
    print()
    print(BOLD + "category counts:" + RESET)
    for cat in ("hit", "short", "max", "rand"):
        n = int(counts.get(cat, 0))
        color = CATEGORY_COLOR.get(cat, "")
        print(f"  {color}{cat:>6}{RESET}: {n}")

    final_log_prob = df["cum_log_prob"].iloc[-1]
    print()
    print(f"{BOLD}final cum_log_prob:{RESET} {final_log_prob:.3f}"
          f"  (raw weight underflows to 0 well before this for any nontrivial ray count --"
          f" compare cum_log_prob between poses instead of the raw weight)")


def plot_poses(gt_x, gt_y, gt_theta, x, y, theta):
    """Static (single-frame) version of animate_particles.py's map + heading-arrow
    display -- just the two poses, no particle cloud. Position only, not the ray
    fan itself; run this alongside the printed table, not instead of it."""
    import matplotlib.pyplot as plt

    img = plt.imread(str(MAP_PNG))
    h, w = img.shape[0], img.shape[1]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img, extent=[0, w, h, 0])
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")

    ax.quiver(
        [gt_x], [gt_y], [HEADING_ARROW_LEN * math.cos(gt_theta)],
        [HEADING_ARROW_LEN * math.sin(gt_theta)], color="red", angles="xy",
        scale_units="xy", scale=1, width=0.006, zorder=5, label="ground truth",
    )
    ax.quiver(
        [x], [y], [HEADING_ARROW_LEN * math.cos(theta)],
        [HEADING_ARROW_LEN * math.sin(theta)], color="blue", angles="xy",
        scale_units="xy", scale=1, width=0.006, zorder=5, label="probe",
    )
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.set_title(
        f"ground truth: ({gt_x:.0f}, {gt_y:.0f}, {gt_theta:.3f}rad)  |  "
        f"probe: ({x:.0f}, {y:.0f}, {theta:.3f}rad)"
    )
    fig.tight_layout()
    plt.show()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt-x", type=float, required=True, help="ground-truth pose x, map-pixel coords")
    ap.add_argument("--gt-y", type=float, required=True, help="ground-truth pose y, map-pixel coords")
    ap.add_argument("--gt-theta", type=float, required=True, help="ground-truth pose heading, radians")
    ap.add_argument("--x", type=float, required=True, help="probe pose x, map-pixel coords")
    ap.add_argument("--y", type=float, required=True, help="probe pose y, map-pixel coords")
    ap.add_argument("--theta", type=float, required=True, help="probe pose heading, radians")
    ap.add_argument("--rays", type=int, default=60, help="number of LIDAR rays (default 60)")
    ap.add_argument("--plot", action="store_true",
                     help="also open a matplotlib window showing both poses (position + "
                          "heading arrow) on the map. Position only, not the individual "
                          "ray fan -- use the printed table for per-ray detail.")
    args = ap.parse_args()

    build_binary()
    csv_text = run_probe(args.gt_x, args.gt_y, args.gt_theta, args.x, args.y, args.theta, args.rays)
    print_table(csv_text)
    if args.plot:
        plot_poses(args.gt_x, args.gt_y, args.gt_theta, args.x, args.y, args.theta)


if __name__ == "__main__":
    main()
