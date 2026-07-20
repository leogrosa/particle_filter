#include "range_libc/includes/RangeLib.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace ranges;
using Clock = std::chrono::steady_clock;

#define Q(x) #x
#define QUOTE(x) Q(x)

// Same setup as mcl_bench.cpp, restricted to GiantLUTCast (glt) only, so a Jetson rerun
// doesn't have to redo the bl/rm/cddt/pcddt sweeps to add the LUT method's numbers.
static const float MAP_RESOLUTION = 0.0504f;
static const float MAX_RANGE_METERS = 10.0f;
static const int THETA_DISCRETIZATION = 112;
static const double Z_SHORT = 0.01, Z_MAX = 0.07, Z_RAND = 0.12, Z_HIT = 0.75, SIGMA_HIT = 8.0;
static const float MOTION_DISPERSION_X = 0.05f, MOTION_DISPERSION_Y = 0.025f, MOTION_DISPERSION_THETA = 0.25f;

static const int WARMUP_ITERS = 3;
static const int TIMED_ITERS = 10;

#ifdef LUT_CLOCK_SPLIT
// Calls timed per accumulated iteration, for the discretize_theta-vs-memory-access clock split
// below. Large enough that std::chrono::steady_clock::now() overhead (paid twice per iteration,
// not once per call) is negligible next to the loop body itself.
static const int CLOCK_SPLIT_CALLS_PER_ITER = 200000;
static const int CLOCK_SPLIT_POOL = 4096;
#endif

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

static void set_identity_ros_transform(OMap &map) {
    map.world_scale = 1.0f;
    map.world_angle = 0.0f;
    map.world_origin_x = 0.0f;
    map.world_origin_y = 0.0f;
    map.world_sin_angle = 0.0f;
    map.world_cos_angle = 1.0f;
}

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

#ifdef LUT_CLOCK_SPLIT
// Faithful standalone replica of GiantLUTCast::discretize_theta (RangeLib.h, ~line 1845), for
// the exact branch combination this project actually compiles with (RangeLib.h lines 66-68,
// not overridden anywhere in bench/CMakeLists.txt -- verified before writing this):
// _USE_ALTERNATE_MOD=1 (while-loop wrap, not fmod), _USE_CACHED_CONSTANTS=1 (precomputed
// theta_discretization/M_2PI), _USE_FAST_ROUND=0 (roundf, not the "+0.5, truncate" trick).
//
// Reproduced here rather than called from RangeLib.h because discretize_theta is protected and
// giant_lut is a private implementation detail of GiantLUTCast -- neither should be reached into
// from outside the class. This function needs only THETA_DISCRETIZATION (line 19 above), the
// same value passed to GiantLUTCast's constructor in main(), so it needs no access to the object
// at all: it is pure arithmetic, with no memory indirection through giant_lut.
static int discretize_theta_replica(float theta) {
    static const double M_2PI_REPLICA = 6.28318530718;  // RangeLib.h line 61, copied verbatim
    // mirrors GiantLUTCast's cached member (RangeLib.h lines 1795-1796): int/double divide,
    // then narrow to float once, at "construction" time -- not recomputed per call.
    static const float THETA_DISC_DIV_M2PI = (float)((double)THETA_DISCRETIZATION / M_2PI_REPLICA);

    if (theta < 0.0) {
        while (theta < 0.0) theta += M_2PI_REPLICA;
    } else if (theta > M_2PI_REPLICA) {
        while (theta > M_2PI_REPLICA) theta -= M_2PI_REPLICA;
    }
    int rounded = (int)roundf(theta * THETA_DISC_DIV_M2PI);
    int binned = rounded % THETA_DISCRETIZATION;
    return binned;
}

// Splits calc_range's time into discretize_theta (arithmetic-only, via the replica above) vs.
// the giant_lut[x][y][theta] access (memory), by timing many calls of each separately and
// subtracting. See the printed NOTE below for why only the ratio, not the raw ns/call figures,
// should be trusted.
static void run_lut_clock_split(GiantLUTCast &glt, OMap &map, std::mt19937 &rng) {
    std::uniform_real_distribution<float> heading_dist(0.0f, 2.0f * (float)M_PI);
    std::vector<float> pool_x(CLOCK_SPLIT_POOL), pool_y(CLOCK_SPLIT_POOL), pool_heading(CLOCK_SPLIT_POOL);
    std::vector<float> one(3);
    for (int i = 0; i < CLOCK_SPLIT_POOL; ++i) {
        sample_free_particles(map, one, 1, rng);
        pool_x[i] = one[0];
        pool_y[i] = one[1];
        pool_heading[i] = heading_dist(rng);
    }

    // volatile sinks: force the compiler to actually keep both loops (a pure discard would let
    // it optimize the arithmetic-only loop away entirely, since discretize_theta_replica has no
    // observable side effect otherwise).
    volatile int sink_i = 0;
    volatile float sink_f = 0.0f;
    double sum_arith_ms = 0.0, sum_total_ms = 0.0;

    for (int iter = 0; iter < WARMUP_ITERS + TIMED_ITERS; ++iter) {
        auto t0 = Clock::now();
        int acc_i = 0;
        for (int i = 0; i < CLOCK_SPLIT_CALLS_PER_ITER; ++i) {
            acc_i += discretize_theta_replica(pool_heading[i % CLOCK_SPLIT_POOL]);
        }
        auto t1 = Clock::now();
        sink_i = acc_i;

        float acc_f = 0.0f;
        for (int i = 0; i < CLOCK_SPLIT_CALLS_PER_ITER; ++i) {
            int j = i % CLOCK_SPLIT_POOL;
            // real, public, unmodified GiantLUTCast::calc_range -- arithmetic (discretize_theta)
            // + the giant_lut[x][y][theta] memory access it does internally.
            acc_f += glt.calc_range(pool_x[j], pool_y[j], pool_heading[j]);
        }
        auto t2 = Clock::now();
        sink_f = acc_f;

        if (iter >= WARMUP_ITERS) {
            sum_arith_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
            sum_total_ms += std::chrono::duration<double, std::milli>(t2 - t1).count();
        }
    }
    (void)sink_i;
    (void)sink_f;

    double t_arith_ms = sum_arith_ms / TIMED_ITERS;
    double t_total_ms = sum_total_ms / TIMED_ITERS;
    double t_mem_ms = t_total_ms - t_arith_ms;
    long long calls = (long long)CLOCK_SPLIT_CALLS_PER_ITER;

    std::printf("\n=== LUT_CLOCK_SPLIT: discretize_theta (arithmetic) vs giant_lut access (memory) ===\n");
    std::printf(
        "NOTE: absolute ns/call figures below are inflated by per-call std::chrono::steady_clock\n"
        "overhead (tens of ns), which is on the same order as a single glt.calc_range query\n"
        "(~49ns measured previously on this map/config). That overhead lands ~equally on both\n"
        "accumulated sums, so treat only the RATIO between t_arith and t_total as trustworthy --\n"
        "the raw ns/call numbers below are NOT ground truth.\n");
    std::printf("calls_per_iter=%lld, timed_iters=%d (sum-then-average; %d warmup iters discarded)\n", calls,
                TIMED_ITERS, WARMUP_ITERS);
    std::printf("t_arith (discretize_theta_replica only) : %10.4f ms total  (%.2f ns/call)\n", t_arith_ms,
                t_arith_ms * 1e6 / calls);
    std::printf("t_total (glt.calc_range, real, public)  : %10.4f ms total  (%.2f ns/call)\n", t_total_ms,
                t_total_ms * 1e6 / calls);
    std::printf("t_mem_est = t_total - t_arith            : %10.4f ms total  (%.2f ns/call)\n", t_mem_ms,
                t_mem_ms * 1e6 / calls);
    std::printf("ratio t_arith / t_total   (arithmetic share) = %.4f\n", t_arith_ms / t_total_ms);
    std::printf("ratio t_mem_est / t_total (memory share)     = %.4f\n", t_mem_ms / t_total_ms);
    std::printf("======================================================================================\n\n");
}
#endif

struct PhaseTimes {
    double resample_ms = 0, motion_ms = 0, range_sensor_ms = 0, normalize_ms = 0, total_ms = 0;
};

template <class RangeT>
static PhaseTimes bench_point(RangeT &range_method, OMap &map, int max_particles, std::vector<float> &angles,
                               std::mt19937 &rng) {
    const int num_rays = (int)angles.size();

    std::vector<float> particles(max_particles * 3);
    std::vector<double> weights(max_particles, 1.0 / max_particles);
    std::vector<float> proposal(max_particles * 3);
    std::vector<double> new_weights(max_particles);
    std::vector<int> proposal_indices(max_particles);

    sample_free_particles(map, particles, max_particles, rng);

    std::vector<float> true_pose(3);
    sample_free_particles(map, true_pose, 1, rng);
    std::vector<float> obs(num_rays);
    for (int a = 0; a < num_rays; ++a)
        obs[a] = range_method.calc_range(true_pose[0], true_pose[1], true_pose[2] + angles[a]);

    std::normal_distribution<float> noise_x(0.0f, MOTION_DISPERSION_X);
    std::normal_distribution<float> noise_y(0.0f, MOTION_DISPERSION_Y);
    std::normal_distribution<float> noise_theta(0.0f, MOTION_DISPERSION_THETA);

    PhaseTimes sum;
    for (int iter = 0; iter < WARMUP_ITERS + TIMED_ITERS; ++iter) {
        auto t0 = Clock::now();

        std::discrete_distribution<int> resample_dist(weights.begin(), weights.end());
        for (int i = 0; i < max_particles; ++i) proposal_indices[i] = resample_dist(rng);
        for (int i = 0; i < max_particles; ++i) {
            int j = proposal_indices[i];
            proposal[i * 3 + 0] = particles[j * 3 + 0];
            proposal[i * 3 + 1] = particles[j * 3 + 1];
            proposal[i * 3 + 2] = particles[j * 3 + 2];
        }
        auto t1 = Clock::now();

        for (int i = 0; i < max_particles; ++i) {
            proposal[i * 3 + 0] += noise_x(rng);
            proposal[i * 3 + 1] += noise_y(rng);
            proposal[i * 3 + 2] += noise_theta(rng);
        }
        auto t2 = Clock::now();

        range_method.calc_range_repeat_angles_eval_sensor_model(proposal.data(), angles.data(), obs.data(),
                                                                 new_weights.data(), max_particles, num_rays);
        auto t3 = Clock::now();

        double wsum = 0.0;
        for (int i = 0; i < max_particles; ++i) wsum += new_weights[i];
        for (int i = 0; i < max_particles; ++i) weights[i] = new_weights[i] / wsum;
        particles.swap(proposal);
        auto t4 = Clock::now();

        if (iter >= WARMUP_ITERS) {
            sum.resample_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
            sum.motion_ms += std::chrono::duration<double, std::milli>(t2 - t1).count();
            sum.range_sensor_ms += std::chrono::duration<double, std::milli>(t3 - t2).count();
            sum.normalize_ms += std::chrono::duration<double, std::milli>(t4 - t3).count();
            sum.total_ms += std::chrono::duration<double, std::milli>(t4 - t0).count();
        }
    }

    PhaseTimes avg;
    avg.resample_ms = sum.resample_ms / TIMED_ITERS;
    avg.motion_ms = sum.motion_ms / TIMED_ITERS;
    avg.range_sensor_ms = sum.range_sensor_ms / TIMED_ITERS;
    avg.normalize_ms = sum.normalize_ms / TIMED_ITERS;
    avg.total_ms = sum.total_ms / TIMED_ITERS;
    return avg;
}

template <class RangeT>
static void run_sweep(const char *method_name, RangeT &range_method, OMap &map, const std::vector<int> &particle_counts,
                       std::vector<float> &angles, std::mt19937 &rng) {
    for (int n : particle_counts) {
        PhaseTimes t = bench_point(range_method, map, n, angles, rng);
        double iters_per_sec = 1000.0 / t.total_ms;
        std::printf("%s,%d,%d,%.3f,%.4f,%.4f,%.4f,%.4f,%.4f\n", method_name, n, (int)angles.size(), iters_per_sec,
                    t.total_ms, t.resample_ms, t.motion_ms, t.range_sensor_ms, t.normalize_ms);
    }
}

static std::vector<float> make_angles(int n) {
    std::vector<float> a(n);
    const float min_angle = -3.0f * (float)M_PI / 4.0f, max_angle = 3.0f * (float)M_PI / 4.0f;
    for (int i = 0; i < n; ++i) a[i] = min_angle + (max_angle - min_angle) * (float)i / (float)(n - 1);
    return a;
}

int main() {
    OMap map(QUOTE(MAP_PATH));
    if (map.error()) {
        std::fprintf(stderr, "failed to load map: %s\n", QUOTE(MAP_PATH));
        return 1;
    }
    set_identity_ros_transform(map);

    const int max_range_px = (int)(MAX_RANGE_METERS / MAP_RESOLUTION);
    const int table_width = max_range_px + 1;
    std::vector<double> sensor_table = build_sensor_model_table(table_width);

    // GiantLUTCast's own init (building the full LUT) is slow (~64s per the RangeLibc doc) —
    // that's a one-time cost paid here, not per-iteration, so it doesn't affect the timed sweep.
    GiantLUTCast glt(map, max_range_px, THETA_DISCRETIZATION);
    glt.set_sensor_model(sensor_table.data(), table_width);

    std::mt19937 rng(42);

#ifdef LUT_CLOCK_SPLIT
    run_lut_clock_split(glt, map, rng);
#endif

    std::vector<float> angles_60 = make_angles(60);
    std::vector<float> angles_1080 = make_angles(1080);

    const std::vector<int> sweep_60 = {500, 1000, 2000, 4000, 8000, 11700, 16000, 24000, 32000, 50000, 75000, 100000};
    const std::vector<int> sweep_1080 = {100, 300, 650, 1000, 2000};

    std::printf("method,max_particles,num_rays,iters_per_sec,ms_total,ms_resample,ms_motion,ms_range_sensor,ms_normalize\n");

    run_sweep("glt", glt, map, sweep_60, angles_60, rng);
    run_sweep("glt", glt, map, sweep_1080, angles_1080, rng);

    return 0;
}
