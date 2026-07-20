#include "range_libc/includes/RangeLib.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

using namespace ranges;
using Clock = std::chrono::steady_clock;

#define Q(x) #x
#define QUOTE(x) Q(x)

// Real defaults, ported from mit-racecar/particle_filter/launch/localize.launch
// and maps/basement_fixed.map.yaml, not invented for this benchmark.
static const float MAP_RESOLUTION = 0.0504f;
static const float MAX_RANGE_METERS = 10.0f;
static const int THETA_DISCRETIZATION = 112;
static const double Z_SHORT = 0.01, Z_MAX = 0.07, Z_RAND = 0.12, Z_HIT = 0.75, SIGMA_HIT = 8.0;
static const float MOTION_DISPERSION_X = 0.05f, MOTION_DISPERSION_Y = 0.025f, MOTION_DISPERSION_THETA = 0.25f;

static const int WARMUP_ITERS = 3;
static const int TIMED_ITERS = 10;

// direct port of ParticleFiler.precompute_sensor_model() from particle_filter.py
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
// directly. Geometric correctness against a real robot frame doesn't matter for a timing
// benchmark, only that synthetic observations and particle queries share the same convention.
static void set_identity_ros_transform(OMap &map) {
	map.world_scale = 1.0f;
	map.world_angle = 0.0f;
	map.world_origin_x = 0.0f;
	map.world_origin_y = 0.0f;
	map.world_sin_angle = 0.0f;
	map.world_cos_angle = 1.0f;
}

// mirrors ParticleFiler.initialize_global(): draw only from free-space cells
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

	// one fixed synthetic observation, ray-cast from a free-space pose. Timing doesn't depend
	// on the observed values, only on exercising the real eval_sensor_model code path.
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

// -3pi/4 to 3pi/4 (270 deg FOV, 1080 rays) approximates a Hokuyo UST-10LX-class scanner,
// matching the 1080-ray real-time budget analysis done earlier; the actual racecar's LIDAR
// FOV wasn't available offline, so this is a stand-in, not a measured value.
static std::vector<float> make_angles(int n) {
	std::vector<float> a(n);
	const float min_angle = -3.0f * (float)M_PI / 4.0f, max_angle = 3.0f * (float)M_PI / 4.0f;
	for (int i = 0; i < n; ++i) a[i] = min_angle + (max_angle - min_angle) * (float)i / (float)(n - 1);
	return a;
}

static void print_usage(const char *prog) {
	std::fprintf(stderr,
		"usage: %s --method NAME --rays N --particles N [--seed N]\n"
		"  --method     one of: bl, rm, cddt, pcddt, glt\n"
		"  --rays       number of LIDAR rays (e.g. 60 or 1080)\n"
		"  --particles  particle count for this single data point\n"
		"  --seed       RNG seed (default 42)\n"
		"\n"
		"Runs exactly one (method, rays, particles) point and prints one CSV row (with\n"
		"header). Previously this binary ran the whole method x rays x particle-count\n"
		"sweep internally in one call, which meant one subprocess invocation (from\n"
		"bench_stats.py) could run for 10+ minutes with no way to bound or skip any\n"
		"single slow point -- RM's 1080-ray sweep hit that wall. Now each invocation is\n"
		"one point, so bench_stats.py --points-mode can apply a timeout per particle\n"
		"count and stop growing a sweep once it's clearly over budget, instead of one\n"
		"global timeout over everything.\n",
		prog);
}

int main(int argc, char **argv) {
	unsigned int seed = 42;
	std::string method_name;
	int num_rays = -1;
	int max_particles = -1;

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
		if (arg == "--seed") seed = (unsigned int)std::atoi(value);
		else if (arg == "--method") method_name = value;
		else if (arg == "--rays") num_rays = std::atoi(value);
		else if (arg == "--particles") max_particles = std::atoi(value);
		else {
			std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
			print_usage(argv[0]);
			return 1;
		}
	}
	if (method_name.empty() || num_rays <= 0 || max_particles <= 0) {
		print_usage(argv[0]);
		return 1;
	}

	OMap map(QUOTE(MAP_PATH));
	if (map.error()) {
		std::fprintf(stderr, "failed to load map: %s\n", QUOTE(MAP_PATH));
		return 1;
	}
	set_identity_ros_transform(map);

	const int max_range_px = (int)(MAX_RANGE_METERS / MAP_RESOLUTION);
	const int table_width = max_range_px + 1;
	std::vector<double> sensor_table = build_sensor_model_table(table_width);

	std::mt19937 rng(seed);
	std::vector<float> angles = make_angles(num_rays);
	const std::vector<int> one_point = {max_particles};

	std::printf("method,max_particles,num_rays,iters_per_sec,ms_total,ms_resample,ms_motion,ms_range_sensor,ms_normalize\n");

	// Only construct the requested method -- a caller sweeping "rm" across many particle counts
	// shouldn't also pay CDDT's trace-table build, PCDDT's prune(), or GiantLUTCast's giant_lut
	// precompute on every single invocation.
	if (method_name == "bl") {
		BresenhamsLine bl(map, max_range_px);
		bl.set_sensor_model(sensor_table.data(), table_width);
		run_sweep("bl", bl, map, one_point, angles, rng);
	} else if (method_name == "rm") {
		RayMarching rm(map, max_range_px);
		rm.set_sensor_model(sensor_table.data(), table_width);
		run_sweep("rm", rm, map, one_point, angles, rng);
	} else if (method_name == "cddt") {
		CDDTCast cddt(map, max_range_px, THETA_DISCRETIZATION);
		cddt.set_sensor_model(sensor_table.data(), table_width);
		run_sweep("cddt", cddt, map, one_point, angles, rng);
	} else if (method_name == "pcddt") {
		CDDTCast pcddt(map, max_range_px, THETA_DISCRETIZATION);
		pcddt.prune(max_range_px);
		pcddt.set_sensor_model(sensor_table.data(), table_width);
		run_sweep("pcddt", pcddt, map, one_point, angles, rng);
	} else if (method_name == "glt") {
		GiantLUTCast glt(map, max_range_px, THETA_DISCRETIZATION);
		glt.set_sensor_model(sensor_table.data(), table_width);
		run_sweep("glt", glt, map, one_point, angles, rng);
	} else {
		std::fprintf(stderr, "unknown method: %s (expected bl, rm, cddt, pcddt, or glt)\n", method_name.c_str());
		return 1;
	}

	return 0;
}
