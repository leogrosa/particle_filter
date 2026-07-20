#include "range_libc/includes/RangeLib.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using namespace ranges;
using Clock = std::chrono::steady_clock;

#define Q(x) #x
#define QUOTE(x) Q(x)

// Same setup as mcl_bench.cpp, restricted to RMGPU only, so a Jetson rerun doesn't have to
// redo the bl/rm/cddt/pcddt/glt sweeps to add the GPU method's numbers. Requires -DWITH_CUDA=ON.
static const float MAP_RESOLUTION = 0.0504f;
static const float MAX_RANGE_METERS = 10.0f;
static const int THETA_DISCRETIZATION = 112;
static const double Z_SHORT = 0.01, Z_MAX = 0.07, Z_RAND = 0.12, Z_HIT = 0.75, SIGMA_HIT = 8.0;
static const float MOTION_DISPERSION_X = 0.05f, MOTION_DISPERSION_Y = 0.025f, MOTION_DISPERSION_THETA = 0.25f;

static const int WARMUP_ITERS = 3;
static const int TIMED_ITERS = 10;

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

struct PhaseTimes {
    double resample_ms = 0, motion_ms = 0, range_sensor_ms = 0, normalize_ms = 0, total_ms = 0;
    double gpu_range_ms = 0;  // sub-split of range_sensor_ms: numpy_calc_range_angles (GPU) only,
                              // excludes the CPU sensor-table accumulate step (see bench_point).
};

// RayMarchingGPU::calc_range_repeat_angles_eval_sensor_model is an unimplemented stub upstream --
// it prints "Do not use ... unimplemented" and returns without touching weights (kernels.cu:281-315,
// the real body is commented out and was never finished, including the never-written
// cuda_accumulate_weights kernel). The library's own .pyx wrapper pairs two other, real methods
// instead: numpy_calc_range_angles (a complete GPU kernel, RangeLib.h:855) for ranges, then
// eval_sensor_model (RangeLib.h:533, base RangeMethod class, CPU) for the weight accumulate --
// used here as-is, not reimplemented. gpu_range_ms times only the first call, so it's directly
// comparable to the RangeLibc doc's own 18-28M queries/sec figures (which are raw calc_range /
// numpy_calc_range_angles numbers, not a fused-with-sensor-model rate -- confirmed by reading the
// doc's own Performance Comparison table). range_sensor_ms (gpu_range_ms + the eval_sensor_model
// call) has no equivalent published number to compare against -- it's a first measurement of the
// full pipeline's real cost given the library's own incomplete GPU implementation.
//
// RayMarchingGPU::calc_range is also deliberately disabled (batched-only, see RangeLib.h) -- the
// synthetic observation is generated with a CPU RayMarching instance instead. Timing doesn't
// depend on the observed values, only on exercising the real eval_sensor_model code path (same
// rationale as mcl_bench.cpp's bench_point).
static PhaseTimes bench_point(RayMarchingGPU &range_method, RayMarching &obs_source, OMap &map, int max_particles,
                               std::vector<float> &angles, std::mt19937 &rng) {
    const int num_rays = (int)angles.size();

    std::vector<float> particles(max_particles * 3);
    std::vector<double> weights(max_particles, 1.0 / max_particles);
    std::vector<float> proposal(max_particles * 3);
    std::vector<double> new_weights(max_particles);
    std::vector<int> proposal_indices(max_particles);
    std::vector<float> ranges(max_particles * num_rays);

    sample_free_particles(map, particles, max_particles, rng);

    std::vector<float> true_pose(3);
    sample_free_particles(map, true_pose, 1, rng);
    std::vector<float> obs(num_rays);
    for (int a = 0; a < num_rays; ++a)
        obs[a] = obs_source.calc_range(true_pose[0], true_pose[1], true_pose[2] + angles[a]);

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

        // RayMarchingGPU::numpy_calc_range_angles (range_libc, not modified here) internally
        // splits large calls into CHUNK_SIZE-sized pieces using particles_per_iter =
        // ceil(CHUNK_SIZE/num_rays) -- that ceil can make particles_per_iter*num_rays exceed
        // CHUNK_SIZE (e.g. CHUNK_SIZE=262144, num_rays=60 -> ceil gives 4370, 4370*60=262200),
        // overflowing its device output buffer once it has to split at all (confirmed: a real
        // cudaErrorInvalidValue on Jetson at max_particles=8000, num_rays=60). Calling it here in
        // batches that never force that internal split (each batch's own particle*ray count safely
        // under CHUNK_SIZE) sidesteps the bug without touching range_libc, which the real
        // mit-racecar Python node also depends on.
        const int safe_batch = std::max(1, CHUNK_SIZE / num_rays);
        for (int off = 0; off < max_particles; off += safe_batch) {
            int n = std::min(safe_batch, max_particles - off);
            range_method.numpy_calc_range_angles(&proposal[off * 3], angles.data(), &ranges[off * num_rays], n, num_rays);
        }
        auto t2b = Clock::now();

        // eval_sensor_model lives in the RangeMethod base class (RangeLib.h:533), inherited by
        // RayMarchingGPU unchanged -- CPU-only, uses map.world_scale and the sensor_model member
        // populated by RayMarchingGPU's own set_sensor_model override (RangeLib.h:878). This is
        // the exact function the library's own .pyx Cython wrapper pairs with
        // numpy_calc_range_angles for GPU sensor evaluation -- not a hand-rolled duplicate.
        range_method.eval_sensor_model(obs.data(), ranges.data(), new_weights.data(), num_rays, max_particles);
        auto t3 = Clock::now();

        double wsum = 0.0;
        for (int i = 0; i < max_particles; ++i) wsum += new_weights[i];
        for (int i = 0; i < max_particles; ++i) weights[i] = new_weights[i] / wsum;
        particles.swap(proposal);
        auto t4 = Clock::now();

        if (iter >= WARMUP_ITERS) {
            sum.resample_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
            sum.motion_ms += std::chrono::duration<double, std::milli>(t2 - t1).count();
            sum.gpu_range_ms += std::chrono::duration<double, std::milli>(t2b - t2).count();
            sum.range_sensor_ms += std::chrono::duration<double, std::milli>(t3 - t2).count();
            sum.normalize_ms += std::chrono::duration<double, std::milli>(t4 - t3).count();
            sum.total_ms += std::chrono::duration<double, std::milli>(t4 - t0).count();
        }
    }

    PhaseTimes avg;
    avg.resample_ms = sum.resample_ms / TIMED_ITERS;
    avg.motion_ms = sum.motion_ms / TIMED_ITERS;
    avg.gpu_range_ms = sum.gpu_range_ms / TIMED_ITERS;
    avg.range_sensor_ms = sum.range_sensor_ms / TIMED_ITERS;
    avg.normalize_ms = sum.normalize_ms / TIMED_ITERS;
    avg.total_ms = sum.total_ms / TIMED_ITERS;
    return avg;
}

static void run_sweep(const char *method_name, RayMarchingGPU &range_method, RayMarching &obs_source, OMap &map,
                      const std::vector<int> &particle_counts, std::vector<float> &angles, std::mt19937 &rng) {
    for (int n : particle_counts) {
        PhaseTimes t = bench_point(range_method, obs_source, map, n, angles, rng);
        double iters_per_sec = 1000.0 / t.total_ms;
        std::printf("%s,%d,%d,%.3f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n", method_name, n, (int)angles.size(), iters_per_sec,
                    t.total_ms, t.resample_ms, t.motion_ms, t.range_sensor_ms, t.normalize_ms, t.gpu_range_ms);
    }
}

static std::vector<float> make_angles(int n) {
    std::vector<float> a(n);
    const float min_angle = -3.0f * (float)M_PI / 4.0f, max_angle = 3.0f * (float)M_PI / 4.0f;
    for (int i = 0; i < n; ++i) a[i] = min_angle + (max_angle - min_angle) * (float)i / (float)(n - 1);
    return a;
}

int main(int argc, char **argv) {
    // Seed is a CLI arg (default 42) rather than hardcoded, because RM/RMGPU's per-ray marching
    // step count depends on which particle positions get sampled -- see rmgpu_stats.py, which
    // runs this binary under several different seeds and averages, to separate real particle-count
    // scaling from seed-dependent workload variance (a single seed's sweep can show non-monotonic
    // timing purely from which map regions happened to get sampled at each particle count).
    unsigned int seed = 42;
    if (argc > 1) seed = (unsigned int)std::atoi(argv[1]);

    OMap map(QUOTE(MAP_PATH));
    if (map.error()) {
        std::fprintf(stderr, "failed to load map: %s\n", QUOTE(MAP_PATH));
        return 1;
    }
    set_identity_ros_transform(map);

    const int max_range_px = (int)(MAX_RANGE_METERS / MAP_RESOLUTION);
    const int table_width = max_range_px + 1;
    std::vector<double> sensor_table = build_sensor_model_table(table_width);

    RayMarching rm(map, max_range_px);  // CPU stand-in, used only to generate obs[] (see bench_point)
    rm.set_sensor_model(sensor_table.data(), table_width);

    RayMarchingGPU rmgpu(map, max_range_px);
    rmgpu.set_sensor_model(sensor_table.data(), table_width);

    std::mt19937 rng(seed);

    std::vector<float> angles_60 = make_angles(60);
    std::vector<float> angles_1080 = make_angles(1080);

    const std::vector<int> sweep_60 = {500, 1000, 2000, 4000, 8000, 11700, 16000, 24000, 32000, 50000, 75000, 100000};
    const std::vector<int> sweep_1080 = {100, 300, 650, 1000, 2000};

    std::printf("method,max_particles,num_rays,iters_per_sec,ms_total,ms_resample,ms_motion,ms_range_sensor,ms_normalize,ms_gpu_range\n");

    run_sweep("rmgpu", rmgpu, rm, map, sweep_60, angles_60, rng);
    run_sweep("rmgpu", rmgpu, rm, map, sweep_1080, angles_1080, rng);

    return 0;
}
