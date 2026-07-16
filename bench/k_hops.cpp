#include "range_libc/includes/RangeLib.h"

#include <algorithm>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

using namespace ranges;

#define Q(x) #x
#define QUOTE(x) Q(x)

static const float MAP_RESOLUTION = 0.0504f;
static const float MAX_RANGE_METERS = 10.0f;
static const int NUM_SAMPLES = 100000;

int main() {
	OMap map(QUOTE(MAP_PATH));
	if (map.error()) {
		std::fprintf(stderr, "failed to load map: %s\n", QUOTE(MAP_PATH));
		return 1;
	}

	const int max_range_px = (int)(MAX_RANGE_METERS / MAP_RESOLUTION);
	RayMarching rm(map, max_range_px);

	std::mt19937 rng(1234);
	std::uniform_real_distribution<float> x_dist(1.0f, (float)map.width - 2.0f);
	std::uniform_real_distribution<float> y_dist(1.0f, (float)map.height - 2.0f);
	std::uniform_real_distribution<float> theta_dist(0.0f, 2.0f * (float)M_PI);

	FILE *csv = std::fopen("k_hops_samples.csv", "w");
	if (!csv) {
		std::fprintf(stderr, "failed to open k_hops_samples.csv for writing\n");
		return 1;
	}
	std::fprintf(csv, "x,y,theta,range_px,hops\n");

	// one query = one (free-space particle pose, absolute heading) pair, matching the same
	// free-space rejection sampling mcl_bench.cpp uses for particles. Marginal distribution of
	// k over this query set, not tied to any specific particle/ray-angle structure.
	std::vector<long> hops(NUM_SAMPLES);
	for (int i = 0; i < NUM_SAMPLES; ++i) {
		float x, y;
		do {
			x = x_dist(rng);
			y = y_dist(rng);
		} while (map.get((int)x, (int)y));
		float theta = theta_dist(rng);

		float range_px = rm.calc_range(x, y, theta);
		hops[i] = rm.hop_count;
		std::fprintf(csv, "%.3f,%.3f,%.5f,%.3f,%ld\n", x, y, theta, range_px, hops[i]);
	}
	std::fclose(csv);

	std::vector<long> sorted_hops = hops;
	std::sort(sorted_hops.begin(), sorted_hops.end());
	long sum = std::accumulate(sorted_hops.begin(), sorted_hops.end(), 0L);
	double mean = (double)sum / NUM_SAMPLES;

	std::printf("samples=%d mean_k=%.3f median_k=%ld min_k=%ld max_k=%ld p05_k=%ld p95_k=%ld\n", NUM_SAMPLES, mean,
	            sorted_hops[NUM_SAMPLES / 2], sorted_hops.front(), sorted_hops.back(),
	            sorted_hops[(size_t)(NUM_SAMPLES * 0.05)], sorted_hops[(size_t)(NUM_SAMPLES * 0.95)]);
	std::printf("wrote per-sample data to k_hops_samples.csv\n");
	return 0;
}
