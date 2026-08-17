#pragma once

// Shared constants/helpers extracted from mcl_convergence.cpp (2026-08-17) so
// likelihood_probe.cpp can reuse the exact same sensor-model table and world
// transform instead of re-deriving a second, possibly-drifting copy. Kept to
// just what likelihood_probe.cpp actually needs -- mcl_convergence.cpp's own
// motion-model/resampling/logging machinery stays local to that file, not
// duplicated here.
//
// All `static`, not `inline`: matches this codebase's existing convention
// (every .cpp here already declares its own private copies of these same
// functions with internal linkage) -- each translation unit that includes
// this header gets its own private copy, which is fine for small free
// functions/constants like these and avoids an ODR/inline discussion.

#include "range_libc/includes/RangeLib.h"

#include <cmath>
#include <vector>

using namespace ranges;

static const float MAP_RESOLUTION = 0.0504f;
static const float MAX_RANGE_METERS = 10.0f;
static const int THETA_DISCRETIZATION = 112;
static const double Z_SHORT = 0.01, Z_MAX = 0.07, Z_RAND = 0.12, Z_HIT = 0.75,
                    SIGMA_HIT = 8.0;

// direct port of ParticleFiler.precompute_sensor_model() from
// particle_filter.py, same as mcl_bench.cpp / mcl_bench_lut.cpp /
// mcl_convergence.cpp.
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
// coordinates directly. Same as mcl_bench.cpp / mcl_bench_lut.cpp /
// mcl_convergence.cpp.
static void set_identity_ros_transform(OMap &map) {
  map.world_scale = 1.0f;
  map.world_angle = 0.0f;
  map.world_origin_x = 0.0f;
  map.world_origin_y = 0.0f;
  map.world_sin_angle = 0.0f;
  map.world_cos_angle = 1.0f;
}

// -3pi/4 to 3pi/4 (270 deg FOV), same convention as mcl_bench.cpp /
// mcl_bench_lut.cpp / mcl_convergence.cpp.
static std::vector<float> make_angles(int n) {
  std::vector<float> a(n);
  const float min_angle = -3.0f * (float)M_PI / 4.0f,
              max_angle = 3.0f * (float)M_PI / 4.0f;
  for (int i = 0; i < n; ++i)
    a[i] = min_angle + (max_angle - min_angle) * (float)i / (float)(n - 1);
  return a;
}
