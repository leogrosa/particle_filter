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

// Does particle-filter convergence shrink the memory working set touched by GiantLUTCast's
// giant_lut[(int)x][(int)y][theta_bin] lookups (RangeLib.h ~1881-1892)? The lookup address is the
// PARTICLE's own position, not where its ray terminates -- so a scattered population touches
// essentially-random rows of a ~378MB table (cache-hostile), while a converged one should reuse a
// handful of adjacent rows. mcl_bench.cpp / mcl_bench_lut.cpp average WARMUP+TIMED iterations
// together, which throws away exactly this scattered->converged transition. This binary instead
// runs --iters iterations and logs every one individually. Local-only / exploratory -- NOT a
// replacement for the project's sanctioned Jetson timing sweeps (mcl_bench*.cpp); total
// per-iteration wall time here doesn't need to fit any real-time budget, only the
// calc_range_repeat_angles_eval_sensor_model call itself is timed.
static const float MAP_RESOLUTION = 0.0504f;
static const float MAX_RANGE_METERS = 10.0f;
static const int THETA_DISCRETIZATION = 112;
static const double Z_SHORT = 0.01, Z_MAX = 0.07, Z_RAND = 0.12, Z_HIT = 0.75, SIGMA_HIT = 8.0;
static const float MOTION_DISPERSION_X = 0.05f, MOTION_DISPERSION_Y = 0.025f, MOTION_DISPERSION_THETA = 0.25f;

// direct port of ParticleFiler.precompute_sensor_model() from particle_filter.py, same as
// mcl_bench.cpp / mcl_bench_lut.cpp
static std::vector<double> build_sensor_model_table(int table_width) {
    std::vector<double> table(table_width * table_width);
    for (int d = 0; d < table_width; ++d) {
        double norm = 0.0;
        for (int r = 0; r < table_width; ++r) {
            double prob = 0.0;
            double z = (double)(r - d);
            prob += Z_HIT * std::exp(-(z * z) / (2.0 * SIGMA_HIT * SIGMA_HIT)) / (SIGMA_HIT * std::sqrt(2.0 * M_PI));
            if (r < d) prob += 2.0 * Z_SHORT * (d - r) / (double)d;
            if (r == table_width - 1) prob += Z_MAX;
            if (r < table_width - 1) prob += Z_RAND / (double)(table_width - 1);
            norm += prob;
            table[r * table_width + d] = prob;
        }
        for (int r = 0; r < table_width; ++r) table[r * table_width + d] /= norm;
    }
    return table;
}

// world_scale=1, identity origin/rotation: "world" coordinates equal map pixel coordinates
// directly. Same as mcl_bench.cpp / mcl_bench_lut.cpp.
static void set_identity_ros_transform(OMap &map) {
    map.world_scale = 1.0f;
    map.world_angle = 0.0f;
    map.world_origin_x = 0.0f;
    map.world_origin_y = 0.0f;
    map.world_sin_angle = 0.0f;
    map.world_cos_angle = 1.0f;
}

// mirrors ParticleFiler.initialize_global(): draw only from free-space cells
// TEMPORARILY REVERTED to the pre-fix map.get() check (2026-08-14) to isolate whether the
// raw_grid-threshold fix is the cause of a since-observed, unexplained LUT-build slowdown --
// GiantLUTCast's constructor runs before this function is ever called, so this shouldn't matter,
// but re-testing rather than assuming. Re-apply the raw_grid fix once this is ruled out.
static void sample_free_particles(OMap &map, std::vector<float> &particles, int n, std::mt19937 &rng) {
    std::uniform_real_distribution<float> x_dist(1.0f, (float)map.width - 2.0f);
    std::uniform_real_distribution<float> y_dist(1.0f, (float)map.height - 2.0f);
    std::uniform_real_distribution<float> theta_dist(0.0f, 2.0f * (float)M_PI);
    for (int i = 0; i < n; ++i) {
        float x, y;
        do {
            x = x_dist(rng);
            y = y_dist(rng);
        } while (map.get((int)x, (int)y));
        particles[i * 3 + 0] = x;
        particles[i * 3 + 1] = y;
        particles[i * 3 + 2] = theta_dist(rng);
    }
}

// -3pi/4 to 3pi/4 (270 deg FOV), same convention as mcl_bench.cpp / mcl_bench_lut.cpp.
static std::vector<float> make_angles(int n) {
    std::vector<float> a(n);
    const float min_angle = -3.0f * (float)M_PI / 4.0f, max_angle = 3.0f * (float)M_PI / 4.0f;
    for (int i = 0; i < n; ++i) a[i] = min_angle + (max_angle - min_angle) * (float)i / (float)(n - 1);
    return a;
}

// One row of ground-truth pose (+velocity, currently unused) per PF iteration.
struct Pose {
    float t = 0.0f, x = 0.0f, y = 0.0f, theta = 0.0f, vx = 0.0f, vy = 0.0f;
};

// Reads iter,t,x,y,theta,vx,vy rows (one header line, skipped) from a caller-supplied trajectory
// CSV -- vx,vy are parsed but ignored for now. This file-based path exists so a future phase can
// point this same binary at a real moving trajectory with zero code changes to the per-iteration
// loop in main(): only where the ground-truth pose comes from differs, not how it's used.
static bool load_trajectory_csv(const std::string &path, int iters, std::vector<Pose> &out, std::string &err) {
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
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string tok;
        std::vector<double> vals;
        while (std::getline(ss, tok, ',')) vals.push_back(std::atof(tok.c_str()));
        if (vals.size() < 7) {
            err = "trajectory row has fewer than 7 columns (want iter,t,x,y,theta,vx,vy): " + line;
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

static void print_usage(const char *prog) {
    std::fprintf(stderr,
        "usage: %s --particles N --rays N --iters N --seed N --out-prefix PATH [--trajectory PATH]\n"
        "  --particles   particle count\n"
        "  --rays        number of LIDAR rays (e.g. 60)\n"
        "  --iters       number of PF iterations to run (every iteration logged, no warmup/timed split)\n"
        "  --seed        RNG seed\n"
        "  --out-prefix  output files become <PATH>_timing.csv, <PATH>_particles.csv, <PATH>_trajectory.csv\n"
        "  --trajectory  optional: read ground-truth pose per iteration (iter,t,x,y,theta,vx,vy) from\n"
        "                this CSV instead of generating a stand-still trajectory\n"
        "\n"
        "Runs GiantLUTCast's real MCL hot path (see mcl_bench_lut.cpp) for --iters iterations,\n"
        "logging every iteration individually -- not averaged like mcl_bench*.cpp's WARMUP+TIMED\n"
        "sweeps -- to see whether particle convergence shrinks the giant_lut memory working set.\n"
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
        if (arg == "--particles") max_particles = std::atoi(value);
        else if (arg == "--rays") num_rays = std::atoi(value);
        else if (arg == "--iters") iters = std::atoi(value);
        else if (arg == "--seed") { seed_arg = std::atol(value); have_seed = true; }
        else if (arg == "--out-prefix") out_prefix = value;
        else if (arg == "--trajectory") trajectory_path = value;
        else {
            std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
            print_usage(argv[0]);
            return 1;
        }
    }
    if (max_particles <= 0 || num_rays <= 0 || iters <= 0 || !have_seed || out_prefix.empty()) {
        print_usage(argv[0]);
        return 1;
    }
    unsigned int seed = (unsigned int)seed_arg;

    OMap map(QUOTE(MAP_PATH));
    if (map.error()) {
        std::fprintf(stderr, "failed to load map: %s\n", QUOTE(MAP_PATH));
        return 1;
    }
    set_identity_ros_transform(map);

    const int max_range_px = (int)(MAX_RANGE_METERS / MAP_RESOLUTION);
    const int table_width = max_range_px + 1;
    std::vector<double> sensor_table = build_sensor_model_table(table_width);

    // GiantLUTCast's own init (building the full LUT) is slow -- a one-time cost paid here, not
    // per-iteration, same pattern as mcl_bench_lut.cpp. This tool is LUT-only, not a multi-method
    // sweep like mcl_bench.cpp: the whole point is GiantLUTCast's position-indexed lookup.
    std::printf("Building LUT\n");
    std::fflush(stdout);
    GiantLUTCast glt(map, max_range_px, THETA_DISCRETIZATION);
    std::printf("Built\n");
    std::fflush(stdout);
    glt.set_sensor_model(sensor_table.data(), table_width);

    std::mt19937 rng(seed);
    std::vector<float> angles = make_angles(num_rays);

    // Ground-truth pose per iteration -- either read from --trajectory, or (default) a single
    // free-space pose held constant across every iteration ("stand still"). Either way, output a
    // trajectory.csv below so downstream plotting code always has one to read regardless of mode.
    std::vector<Pose> trajectory;
    if (!trajectory_path.empty()) {
        std::string err;
        if (!load_trajectory_csv(trajectory_path, iters, trajectory, err)) {
            std::fprintf(stderr, "%s\n", err.c_str());
            return 1;
        }
        trajectory.resize(iters);  // ignore any extra rows beyond --iters
    } else {
        std::vector<float> true_pose(3);
        sample_free_particles(map, true_pose, 1, rng);
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

    // Iteration 0 starts from a uniformly-scattered-across-all-free-space population with
    // uniform weights, same as mcl_bench.cpp / mcl_bench_lut.cpp's bench_point().
    std::vector<float> particles(max_particles * 3);
    std::vector<double> weights(max_particles, 1.0 / max_particles);
    sample_free_particles(map, particles, max_particles, rng);

    std::vector<float> proposal(max_particles * 3);
    std::vector<double> new_weights(max_particles);
    std::vector<int> proposal_indices(max_particles);
    std::vector<float> obs(num_rays);

    std::normal_distribution<float> noise_x(0.0f, MOTION_DISPERSION_X);
    std::normal_distribution<float> noise_y(0.0f, MOTION_DISPERSION_Y);
    std::normal_distribution<float> noise_theta(0.0f, MOTION_DISPERSION_THETA);

    std::string timing_path = out_prefix + "_timing.csv";
    std::string particles_path = out_prefix + "_particles.csv";
    std::string trajectory_out_path = out_prefix + "_trajectory.csv";

    FILE *f_timing = std::fopen(timing_path.c_str(), "w");
    FILE *f_particles = std::fopen(particles_path.c_str(), "w");
    FILE *f_trajectory = std::fopen(trajectory_out_path.c_str(), "w");
    if (!f_timing || !f_particles || !f_trajectory) {
        std::fprintf(stderr, "failed to open output files at prefix: %s\n", out_prefix.c_str());
        return 1;
    }

    std::fprintf(f_timing, "iter,distinct_cells,mean_dist_to_true,stddev_x,stddev_y,ms_range_sensor\n");
    std::fprintf(f_particles, "iter,particle_id,x,y,theta,weight\n");
    std::fprintf(f_trajectory, "iter,t,x,y,theta,vx,vy\n");

    for (int iter = 0; iter < iters; ++iter) {
        const Pose &gt = trajectory[iter];
        std::fprintf(f_trajectory, "%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n", iter, gt.t, gt.x, gt.y, gt.theta, gt.vx,
                     gt.vy);

        // 1. resample, weighted by the previous iteration's normalized weights (uniform 1/N at
        // iter 0, from initialization above).
        std::discrete_distribution<int> resample_dist(weights.begin(), weights.end());
        for (int i = 0; i < max_particles; ++i) proposal_indices[i] = resample_dist(rng);
        for (int i = 0; i < max_particles; ++i) {
            int j = proposal_indices[i];
            proposal[i * 3 + 0] = particles[j * 3 + 0];
            proposal[i * 3 + 1] = particles[j * 3 + 1];
            proposal[i * 3 + 2] = particles[j * 3 + 2];
        }

        // 2. motion
        for (int i = 0; i < max_particles; ++i) {
            proposal[i * 3 + 0] += noise_x(rng);
            proposal[i * 3 + 1] += noise_y(rng);
            proposal[i * 3 + 2] += noise_theta(rng);
        }

        // 3. working-set / convergence stats from the post-motion, pre-lookup population -- this
        // is the whole point of this tool: does convergence shrink the set of giant_lut rows
        // about to be touched by the timed call in step 5?
        std::unordered_set<long long> distinct_cell_set;
        double sum_x = 0.0, sum_y = 0.0, sum_dist = 0.0;
        for (int i = 0; i < max_particles; ++i) {
            float x = proposal[i * 3 + 0], y = proposal[i * 3 + 1];
            int ix = (int)x, iy = (int)y;
            distinct_cell_set.insert(((long long)ix << 32) | (unsigned int)iy);
            sum_x += x;
            sum_y += y;
            double dx = (double)x - gt.x, dy = (double)y - gt.y;
            sum_dist += std::sqrt(dx * dx + dy * dy);
        }
        double mean_x = sum_x / max_particles;
        double mean_y = sum_y / max_particles;
        double mean_dist_to_true = sum_dist / max_particles;
        double var_x = 0.0, var_y = 0.0;
        for (int i = 0; i < max_particles; ++i) {
            double dx = (double)proposal[i * 3 + 0] - mean_x, dy = (double)proposal[i * 3 + 1] - mean_y;
            var_x += dx * dx;
            var_y += dy * dy;
        }
        var_x /= max_particles;
        var_y /= max_particles;
        double stddev_x = std::sqrt(var_x);
        double stddev_y = std::sqrt(var_y);
        long distinct_cells = (long)distinct_cell_set.size();

        // 4. this iteration's synthetic observation, ray-cast from the ground-truth pose --
        // untimed, same as mcl_bench.cpp / mcl_bench_lut.cpp's obs setup.
        for (int a = 0; a < num_rays; ++a) obs[a] = glt.calc_range(gt.x, gt.y, gt.theta + angles[a]);

        // 5. timed region: only the lookup + sensor-model eval call itself. Resample/motion/
        // normalize are deliberately not timed -- this tool cares about one thing only.
        auto t0 = Clock::now();
        glt.calc_range_repeat_angles_eval_sensor_model(proposal.data(), angles.data(), obs.data(),
                                                         new_weights.data(), max_particles, num_rays);
        auto t1 = Clock::now();
        double ms_range_sensor = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // 6. normalize
        double wsum = 0.0;
        for (int i = 0; i < max_particles; ++i) wsum += new_weights[i];
        for (int i = 0; i < max_particles; ++i) weights[i] = new_weights[i] / wsum;

        // 7. per-particle log: the population actually evaluated this iteration, paired with its
        // just-computed normalized weight.
        for (int i = 0; i < max_particles; ++i) {
            std::fprintf(f_particles, "%d,%d,%.6f,%.6f,%.6f,%.8g\n", iter, i, proposal[i * 3 + 0],
                         proposal[i * 3 + 1], proposal[i * 3 + 2], weights[i]);
        }

        // 8. per-iteration timing/working-set log
        std::fprintf(f_timing, "%d,%ld,%.6f,%.6f,%.6f,%.6f\n", iter, distinct_cells, mean_dist_to_true, stddev_x,
                     stddev_y, ms_range_sensor);

        // 9. next iteration resamples from this iteration's evaluated population
        particles.swap(proposal);
    }

    std::fclose(f_timing);
    std::fclose(f_particles);
    std::fclose(f_trajectory);

    return 0;
}
