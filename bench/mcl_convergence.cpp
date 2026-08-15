#include "range_libc/includes/RangeLib.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

using namespace ranges;
using Clock = std::chrono::steady_clock;

#define Q(x) #x
#define QUOTE(x) Q(x)

// Does particle-filter convergence shrink the memory working set touched by
// GiantLUTCast's giant_lut[(int)x][(int)y][theta_bin] lookups (RangeLib.h
// ~1881-1892)? The lookup address is the PARTICLE's own position, not where its
// ray terminates -- so a scattered population touches essentially-random rows
// of a ~378MB table (cache-hostile), while a converged one should reuse a
// handful of adjacent rows. mcl_bench.cpp / mcl_bench_lut.cpp average
// WARMUP+TIMED iterations together, which throws away exactly this
// scattered->converged transition. This binary instead runs --iters iterations
// and logs every one individually. Local-only / exploratory -- NOT a
// replacement for the project's sanctioned Jetson timing sweeps
// (mcl_bench*.cpp); total per-iteration wall time here doesn't need to fit any
// real-time budget, only the calc_range_repeat_angles_eval_sensor_model call
// itself is timed.
static const float MAP_RESOLUTION = 0.0504f;
static const float MAX_RANGE_METERS = 10.0f;
static const int THETA_DISCRETIZATION = 112;
static const double Z_SHORT = 0.01, Z_MAX = 0.07, Z_RAND = 0.12, Z_HIT = 0.75,
                    SIGMA_HIT = 8.0;
static const float MOTION_DISPERSION_X = 0.05f, MOTION_DISPERSION_Y = 0.025f,
                   MOTION_DISPERSION_THETA = 0.25f;

// direct port of ParticleFiler.precompute_sensor_model() from
// particle_filter.py, same as mcl_bench.cpp / mcl_bench_lut.cpp
static std::vector<double> build_sensor_model_table(int table_width) {
  std::vector<double> table(table_width * table_width);
  for (int d = 0; d < table_width; ++d) {
    double norm = 0.0;
    for (int r = 0; r < table_width; ++r) {
      double prob = 0.0;
      double z = (double)(r - d);
      prob += Z_HIT * std::exp(-(z * z) / (2.0 * SIGMA_HIT * SIGMA_HIT)) /
              (SIGMA_HIT * std::sqrt(2.0 * M_PI));
      if (r < d)
        prob += 2.0 * Z_SHORT * (d - r) / (double)d;
      if (r == table_width - 1)
        prob += Z_MAX;
      if (r < table_width - 1)
        prob += Z_RAND / (double)(table_width - 1);
      norm += prob;
      table[r * table_width + d] = prob;
    }
    for (int r = 0; r < table_width; ++r)
      table[r * table_width + d] /= norm;
  }
  return table;
}

// world_scale=1, identity origin/rotation: "world" coordinates equal map pixel
// coordinates directly. Same as mcl_bench.cpp / mcl_bench_lut.cpp.
static void set_identity_ros_transform(OMap &map) {
  map.world_scale = 1.0f;
  map.world_angle = 0.0f;
  map.world_origin_x = 0.0f;
  map.world_origin_y = 0.0f;
  map.world_sin_angle = 0.0f;
  map.world_cos_angle = 1.0f;
}

// mirrors ParticleFiler.initialize_global(): draw only from free-space cells.
// NOTE: deliberately does NOT use map.get() -- that checks OMap's occupied grid
// (gray < 128), which is correct for ray-casting but classifies
// unknown/unmapped gray (~204 in this map) as free, same as real free space
// (~255). Real free space and unknown-gray are cleanly separated in
// basement_fixed.png's pixel histogram (255/254: ~273.5K px vs 204: ~1.39M px),
// so check raw_grid directly against a strict near-white cutoff instead.
// (Confirmed 2026-08-14 this isn't the cause of a separate, still-unexplained
// slowdown -- reverting to map.get() didn't fix that -- so reinstating.)
static const float FREE_SPACE_GRAY_MIN = 189.0f;

// Scans the map once for every valid free-space cell. Building this list up
// front and sampling a uniform index into it (below) replaces the old
// reject-until-valid loop: that loop's runtime was unbounded (fine in the
// common case, but a near-empty or empty valid fraction spins forever with no
// error), and continuity within a cell was illusory anyway -- every downstream
// lookup (giant_lut, calc_range) truncates to (int)x,(int)y regardless of the
// sampled fractional part.
static std::vector<std::pair<int, int>> build_free_space_cells(OMap &map) {
  std::vector<std::pair<int, int>> cells;
  for (int x = 1; x <= map.width - 2; ++x) {
    for (int y = 1; y <= map.height - 2; ++y) {
      // std::printf("Cell parameters: x=%d, y=%d, gray=%d\n", x, y,
      //             map.raw_grid[x][y]);
      if (map.raw_grid[x][y] >= FREE_SPACE_GRAY_MIN)
        cells.emplace_back(x, y);
    }
  }
  return cells;
}

static void
sample_free_particles(const std::vector<std::pair<int, int>> &free_cells,
                      std::vector<float> &particles, int n, std::mt19937 &rng) {
  std::uniform_int_distribution<size_t> cell_dist(0, free_cells.size() - 1);
  std::uniform_real_distribution<float> theta_dist(0.0f, 2.0f * (float)M_PI);
  for (int i = 0; i < n; ++i) {
    const auto &cell = free_cells[cell_dist(rng)];
    particles[i * 3 + 0] = (float)cell.first;
    particles[i * 3 + 1] = (float)cell.second;
    particles[i * 3 + 2] = theta_dist(rng);
  }
}

// -3pi/4 to 3pi/4 (270 deg FOV), same convention as mcl_bench.cpp /
// mcl_bench_lut.cpp.
static std::vector<float> make_angles(int n) {
  std::vector<float> a(n);
  const float min_angle = -3.0f * (float)M_PI / 4.0f,
              max_angle = 3.0f * (float)M_PI / 4.0f;
  for (int i = 0; i < n; ++i)
    a[i] = min_angle + (max_angle - min_angle) * (float)i / (float)(n - 1);
  return a;
}

// One row of ground-truth pose (+velocity, currently unused) per PF iteration.
struct Pose {
  float t = 0.0f, x = 0.0f, y = 0.0f, theta = 0.0f, vx = 0.0f, vy = 0.0f;
};

// Reads iter,t,x,y,theta,vx,vy rows (one header line, skipped) from a
// caller-supplied trajectory CSV -- vx,vy are parsed but ignored for now. This
// file-based path exists so a future phase can point this same binary at a real
// moving trajectory with zero code changes to the per-iteration loop in main():
// only where the ground-truth pose comes from differs, not how it's used.
static bool load_trajectory_csv(const std::string &path, int iters,
                                std::vector<Pose> &out, std::string &err) {
  std::ifstream f(path);
  if (!f.is_open()) {
    err = "failed to open trajectory file: " + path;
    return false;
  }
  std::string line;
  if (!std::getline(f, line)) {
    err = "trajectory file is empty (expected a header row): " + path;
    return false;
  }
  out.clear();
  while (std::getline(f, line)) {
    if (line.empty())
      continue;
    std::stringstream ss(line);
    std::string tok;
    std::vector<double> vals;
    while (std::getline(ss, tok, ','))
      vals.push_back(std::atof(tok.c_str()));
    if (vals.size() < 7) {
      err = "trajectory row has fewer than 7 columns (want "
            "iter,t,x,y,theta,vx,vy): " +
            line;
      return false;
    }
    Pose p;
    p.t = (float)vals[1];
    p.x = (float)vals[2];
    p.y = (float)vals[3];
    p.theta = (float)vals[4];
    p.vx = (float)vals[5];
    p.vy = (float)vals[6];
    out.push_back(p);
  }
  if ((int)out.size() < iters) {
    err = "trajectory file has " + std::to_string(out.size()) +
          " data rows, need at least --iters=" + std::to_string(iters);
    return false;
  }
  return true;
}

// ---- Per-iteration steps, in the order main() calls them
// ----------------------------------
//
// Loop order is predict -> update -> resample (not the naive resample ->
// predict -> update): resampling is driven by weights computed from an actual
// measurement, so it happens AFTER this iteration's measurement update, using
// the weights it just produced -- not before, from whatever weights the
// previous iteration left behind. At iter 0 that previous state is the initial
// uniform 1/N weighting, and resampling from uniform weights before any
// measurement has happened is pure noise with no information behind it, so the
// old resample-first ordering paid for one entirely useless resample at the
// start of the run. Under this ordering, only the last iteration has no next
// iteration to feed, so main() skips its resample instead.

// Step: predict. Applies independent Gaussian process noise to each particle's
// pose in place.
static void motion_step(std::vector<float> &particles, int n, std::mt19937 &rng,
                        std::normal_distribution<float> &noise_x,
                        std::normal_distribution<float> &noise_y,
                        std::normal_distribution<float> &noise_theta) {
  for (int i = 0; i < n; ++i) {
    particles[i * 3 + 0] += noise_x(rng);
    particles[i * 3 + 1] += noise_y(rng);
    particles[i * 3 + 2] += noise_theta(rng);
  }
}

// Step: working-set / convergence stats over the current (post-motion,
// pre-lookup) population -- the whole point of this tool: does convergence
// shrink the set of giant_lut rows about to be touched by the timed
// measurement_update() call?
static void compute_working_set_stats(const std::vector<float> &particles,
                                      int n, const Pose &gt,
                                      long &distinct_cells,
                                      double &mean_dist_to_true,
                                      double &stddev_x, double &stddev_y) {
  std::unordered_set<long long> distinct_cell_set;
  double sum_x = 0.0, sum_y = 0.0, sum_dist = 0.0;
  for (int i = 0; i < n; ++i) {
    float x = particles[i * 3 + 0], y = particles[i * 3 + 1];
    int ix = (int)x, iy = (int)y;
    distinct_cell_set.insert(((long long)ix << 32) | (unsigned int)iy);
    sum_x += x;
    sum_y += y;
    double dx = (double)x - gt.x, dy = (double)y - gt.y;
    sum_dist += std::sqrt(dx * dx + dy * dy);
  }
  double mean_x = sum_x / n;
  double mean_y = sum_y / n;
  mean_dist_to_true = sum_dist / n;
  double var_x = 0.0, var_y = 0.0;
  for (int i = 0; i < n; ++i) {
    double dx = (double)particles[i * 3 + 0] - mean_x,
           dy = (double)particles[i * 3 + 1] - mean_y;
    var_x += dx * dx;
    var_y += dy * dy;
  }
  var_x /= n;
  var_y /= n;
  stddev_x = std::sqrt(var_x);
  stddev_y = std::sqrt(var_y);
  distinct_cells = (long)distinct_cell_set.size();
}

// Step: this iteration's synthetic observation, ray-cast from the ground-truth
// pose. Untimed, same as mcl_bench.cpp / mcl_bench_lut.cpp's obs setup.
static void build_ground_truth_observation(GiantLUTCast &glt, const Pose &gt,
                                           std::vector<float> &angles,
                                           std::vector<float> &obs,
                                           int num_rays) {
  for (int a = 0; a < num_rays; ++a)
    obs[a] = glt.calc_range(gt.x, gt.y, gt.theta + angles[a]);
}

// Step: update. The one timed region -- lookup + sensor-model eval against
// every particle. Resample/motion/normalize are deliberately not timed -- this
// tool cares about one thing only.
static double
measurement_update(GiantLUTCast &glt, std::vector<float> &particles,
                   std::vector<float> &angles, std::vector<float> &obs,
                   std::vector<double> &new_weights, int n, int num_rays) {
  auto t0 = Clock::now();
  glt.calc_range_repeat_angles_eval_sensor_model(
      particles.data(), angles.data(), obs.data(), new_weights.data(), n,
      num_rays);
  auto t1 = Clock::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// Step: normalize this iteration's raw weights into a proper distribution for
// resampling/logging.
static void normalize_weights(const std::vector<double> &new_weights,
                              std::vector<double> &weights, int n) {
  double wsum = 0.0;
  for (int i = 0; i < n; ++i)
    wsum += new_weights[i];
  for (int i = 0; i < n; ++i)
    weights[i] = new_weights[i] / wsum;
}

// Step: resample, weighted by this iteration's just-normalized weights --
// produces the population the *next* iteration's motion step will start from.
static void resample_step(const std::vector<float> &particles,
                          const std::vector<double> &weights,
                          std::vector<float> &proposal,
                          std::vector<int> &proposal_indices, int n,
                          std::mt19937 &rng) {
  std::discrete_distribution<int> resample_dist(weights.begin(), weights.end());
  for (int i = 0; i < n; ++i)
    proposal_indices[i] = resample_dist(rng);
  for (int i = 0; i < n; ++i) {
    int j = proposal_indices[i];
    proposal[i * 3 + 0] = particles[j * 3 + 0];
    proposal[i * 3 + 1] = particles[j * 3 + 1];
    proposal[i * 3 + 2] = particles[j * 3 + 2];
  }
}

static void log_trajectory_row(FILE *f, int iter, const Pose &gt) {
  std::fprintf(f, "%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n", iter, gt.t, gt.x, gt.y,
               gt.theta, gt.vx, gt.vy);
}

// Logs the population actually evaluated this iteration, paired with its
// just-computed normalized weight (pre-resample -- this iteration's resample,
// if any, is a separate step).
static void log_particles(FILE *f, int iter,
                          const std::vector<float> &particles,
                          const std::vector<double> &weights, int n) {
  for (int i = 0; i < n; ++i) {
    std::fprintf(f, "%d,%d,%.6f,%.6f,%.6f,%.8g\n", iter, i,
                 particles[i * 3 + 0], particles[i * 3 + 1],
                 particles[i * 3 + 2], weights[i]);
  }
}

static void log_timing_row(FILE *f, int iter, long distinct_cells,
                           double mean_dist_to_true, double stddev_x,
                           double stddev_y, double ms_range_sensor) {
  std::fprintf(f, "%d,%ld,%.6f,%.6f,%.6f,%.6f\n", iter, distinct_cells,
               mean_dist_to_true, stddev_x, stddev_y, ms_range_sensor);
}

static void print_usage(const char *prog) {
  std::fprintf(
      stderr,
      "usage: %s --particles N --rays N --iters N --seed N --out-prefix PATH "
      "[--trajectory PATH]\n"
      "  --particles   particle count\n"
      "  --rays        number of LIDAR rays (e.g. 60)\n"
      "  --iters       number of PF iterations to run (every iteration logged, "
      "no warmup/timed split)\n"
      "  --seed        RNG seed\n"
      "  --out-prefix  output directory; created if missing. Writes "
      "<PATH>/timing.csv, "
      "<PATH>/particles.csv, <PATH>/trajectory.csv\n"
      "  --trajectory  optional: read ground-truth pose per iteration "
      "(iter,t,x,y,theta,vx,vy) from\n"
      "                this CSV instead of generating a stand-still "
      "trajectory\n"
      "\n"
      "Runs GiantLUTCast's real MCL hot path (see mcl_bench_lut.cpp) for "
      "--iters iterations,\n"
      "logging every iteration individually -- not averaged like "
      "mcl_bench*.cpp's WARMUP+TIMED\n"
      "sweeps -- to see whether particle convergence shrinks the giant_lut "
      "memory working set.\n"
      "Local-only / exploratory: not the sanctioned Jetson timing sweep.\n",
      prog);
}

int main(int argc, char **argv) {
  int max_particles = -1;
  int num_rays = -1;
  int iters = -1;
  long seed_arg = 0;
  bool have_seed = false;
  std::string out_prefix;
  std::string trajectory_path;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "-h" || arg == "--help") {
      print_usage(argv[0]);
      return 0;
    }
    if (i + 1 >= argc) {
      std::fprintf(stderr, "%s requires a value\n", arg.c_str());
      print_usage(argv[0]);
      return 1;
    }
    const char *value = argv[++i];
    if (arg == "--particles")
      max_particles = std::atoi(value);
    else if (arg == "--rays")
      num_rays = std::atoi(value);
    else if (arg == "--iters")
      iters = std::atoi(value);
    else if (arg == "--seed") {
      seed_arg = std::atol(value);
      have_seed = true;
    } else if (arg == "--out-prefix")
      out_prefix = value;
    else if (arg == "--trajectory")
      trajectory_path = value;
    else {
      std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
      print_usage(argv[0]);
      return 1;
    }
  }
  if (max_particles <= 0 || num_rays <= 0 || iters <= 0 || !have_seed ||
      out_prefix.empty()) {
    print_usage(argv[0]);
    return 1;
  }
  unsigned int seed = (unsigned int)seed_arg;

  std::printf("[setup] loading map\n");
  std::fflush(stdout);
  OMap map(QUOTE(MAP_PATH));
  if (map.error()) {
    std::fprintf(stderr, "failed to load map: %s\n", QUOTE(MAP_PATH));
    return 1;
  }
  set_identity_ros_transform(map);
  std::printf("[setup] map loaded: %dx%d\n", map.width, map.height);
  std::fflush(stdout);

  std::printf("[setup] scanning map for free-space cells\n");
  std::fflush(stdout);
  std::vector<std::pair<int, int>> free_cells = build_free_space_cells(map);
  long total_cells = (long)(map.width - 2) * (long)(map.height - 2);
  std::printf(
      "[setup] found %zu free-space cells out of %ld scanned (%.1f%%)\n",
      free_cells.size(), total_cells,
      total_cells > 0 ? 100.0 * (double)free_cells.size() / (double)total_cells
                      : 0.0);
  std::fflush(stdout);
  if (free_cells.empty()) {
    std::fprintf(stderr, "no free-space cells found in map -- check "
                         "FREE_SPACE_GRAY_MIN / map contents\n");
    return 1;
  }

  const int max_range_px = (int)(MAX_RANGE_METERS / MAP_RESOLUTION);
  const int table_width = max_range_px + 1;
  std::printf("[setup] building sensor model table (width=%d)\n", table_width);
  std::fflush(stdout);
  std::vector<double> sensor_table = build_sensor_model_table(table_width);
  std::printf("[setup] sensor model table built\n");
  std::fflush(stdout);

  // GiantLUTCast's own init (building the full LUT) is slow -- a one-time cost
  // paid here, not per-iteration, same pattern as mcl_bench_lut.cpp. This tool
  // is LUT-only, not a multi-method sweep like mcl_bench.cpp: the whole point
  // is GiantLUTCast's position-indexed lookup.
  std::printf("[setup] building LUT (this is the slow one-time step)\n");
  std::fflush(stdout);
  GiantLUTCast glt(map, max_range_px, THETA_DISCRETIZATION);
  std::printf("[setup] LUT built\n");
  std::fflush(stdout);
  glt.set_sensor_model(sensor_table.data(), table_width);
  std::printf("[setup] sensor model attached to LUT caster\n");
  std::fflush(stdout);

  std::mt19937 rng(seed);
  std::vector<float> angles = make_angles(num_rays);

  // Ground-truth pose per iteration -- either read from --trajectory, or
  // (default) a single free-space pose held constant across every iteration
  // ("stand still"). Either way, output a trajectory.csv below so downstream
  // plotting code always has one to read regardless of mode.
  std::vector<Pose> trajectory;
  if (!trajectory_path.empty()) {
    std::printf("[setup] loading trajectory from %s\n",
                trajectory_path.c_str());
    std::fflush(stdout);
    std::string err;
    if (!load_trajectory_csv(trajectory_path, iters, trajectory, err)) {
      std::fprintf(stderr, "%s\n", err.c_str());
      return 1;
    }
    trajectory.resize(iters); // ignore any extra rows beyond --iters
  } else {
    std::printf("[setup] sampling a stand-still ground-truth pose\n");
    std::fflush(stdout);
    std::vector<float> true_pose(3);
    sample_free_particles(free_cells, true_pose, 1, rng);
    trajectory.assign(iters, Pose());
    for (int i = 0; i < iters; ++i) {
      trajectory[i].t = (float)i;
      trajectory[i].x = true_pose[0];
      trajectory[i].y = true_pose[1];
      trajectory[i].theta = true_pose[2];
      trajectory[i].vx = 0.0f;
      trajectory[i].vy = 0.0f;
    }
  }
  std::printf("[setup] ground-truth pose (iter 0): x=%.3f y=%.3f theta=%.3f\n",
              trajectory[0].x, trajectory[0].y, trajectory[0].theta);
  std::fflush(stdout);

  // Iteration 0 starts from a uniformly-scattered-across-all-free-space
  // population with uniform weights, same as mcl_bench.cpp /
  // mcl_bench_lut.cpp's bench_point().
  std::printf("[setup] sampling %d initial free-space particles\n",
              max_particles);
  std::fflush(stdout);
  std::vector<float> particles(max_particles * 3);
  std::vector<double> weights(max_particles);
  sample_free_particles(free_cells, particles, max_particles, rng);
  std::printf("[setup] initial particles sampled\n");
  std::fflush(stdout);

  std::vector<float> proposal(max_particles * 3);
  std::vector<double> new_weights(max_particles);
  std::vector<int> proposal_indices(max_particles);
  std::vector<float> obs(num_rays);

  std::normal_distribution<float> noise_x(0.0f, MOTION_DISPERSION_X);
  std::normal_distribution<float> noise_y(0.0f, MOTION_DISPERSION_Y);
  std::normal_distribution<float> noise_theta(0.0f, MOTION_DISPERSION_THETA);

  // out_prefix is a directory now (one per run, e.g.
  // bench/results/2026-08-14_153045/), holding plainly-named
  // timing.csv/particles.csv/trajectory.csv -- mkdir -p it in case this binary
  // is invoked directly rather than through run_convergence.py, which also
  // creates it.
  std::system(("mkdir -p " + out_prefix).c_str());
  std::string timing_path = out_prefix + "/timing.csv";
  std::string particles_path = out_prefix + "/particles.csv";
  std::string trajectory_out_path = out_prefix + "/trajectory.csv";

  FILE *f_timing = std::fopen(timing_path.c_str(), "w");
  FILE *f_particles = std::fopen(particles_path.c_str(), "w");
  FILE *f_trajectory = std::fopen(trajectory_out_path.c_str(), "w");
  if (!f_timing || !f_particles || !f_trajectory) {
    std::fprintf(stderr, "failed to open output files at prefix: %s\n",
                 out_prefix.c_str());
    return 1;
  }

  std::fprintf(f_timing, "iter,distinct_cells,mean_dist_to_true,stddev_x,"
                         "stddev_y,ms_range_sensor\n");
  std::fprintf(f_particles, "iter,particle_id,x,y,theta,weight\n");
  std::fprintf(f_trajectory, "iter,t,x,y,theta,vx,vy\n");

  std::printf("[run] starting %d iterations (%d particles, %d rays)\n", iters,
              max_particles, num_rays);
  std::fflush(stdout);

  for (int iter = 0; iter < iters; ++iter) {
    const Pose &gt = trajectory[iter];
    log_trajectory_row(f_trajectory, iter, gt);

    std::printf("[iter %d/%d] motion: applying process noise to %d particles\n",
                iter, iters - 1, max_particles);
    std::fflush(stdout);
    motion_step(particles, max_particles, rng, noise_x, noise_y, noise_theta);

    long distinct_cells;
    double mean_dist_to_true, stddev_x, stddev_y;
    compute_working_set_stats(particles, max_particles, gt, distinct_cells,
                              mean_dist_to_true, stddev_x, stddev_y);
    std::printf("[iter %d/%d] stats: distinct_cells=%ld mean_dist_to_true=%.3f "
                "stddev_x=%.3f stddev_y=%.3f\n",
                iter, iters - 1, distinct_cells, mean_dist_to_true, stddev_x,
                stddev_y);
    std::fflush(stdout);

    build_ground_truth_observation(glt, gt, angles, obs, num_rays);
    std::printf("[iter %d/%d] observation: cast %d ground-truth rays\n", iter,
                iters - 1, num_rays);
    std::fflush(stdout);

    double ms_range_sensor = measurement_update(
        glt, particles, angles, obs, new_weights, max_particles, num_rays);
    std::printf("[iter %d/%d] measurement update: %.3f ms\n", iter, iters - 1,
                ms_range_sensor);
    std::fflush(stdout);

    normalize_weights(new_weights, weights, max_particles);
    std::printf("[iter %d/%d] normalized weights\n", iter, iters - 1);
    std::fflush(stdout);

    log_particles(f_particles, iter, particles, weights, max_particles);
    log_timing_row(f_timing, iter, distinct_cells, mean_dist_to_true, stddev_x,
                   stddev_y, ms_range_sensor);
    std::printf("[iter %d/%d] logged particles + timing rows\n", iter,
                iters - 1);
    std::fflush(stdout);

    // Skip the last iteration's resample -- nothing downstream would ever read
    // it.
    if (iter + 1 < iters) {
      resample_step(particles, weights, proposal, proposal_indices,
                    max_particles, rng);
      particles.swap(proposal);
      std::printf(
          "[iter %d/%d] resample: drew %d particles for the next iteration\n",
          iter, iters - 1, max_particles);
    } else {
      std::printf("[iter %d/%d] last iteration -- skipping resample\n", iter,
                  iters - 1);
    }
    std::fflush(stdout);
  }

  std::printf("[run] done, closing output files\n");
  std::fflush(stdout);

  std::fclose(f_timing);
  std::fclose(f_particles);
  std::fclose(f_trajectory);

  return 0;
}
