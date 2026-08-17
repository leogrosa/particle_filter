#include "range_libc/includes/RangeLib.h"
#include "mcl_common.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

using namespace ranges;

#define Q(x) #x
#define QUOTE(x) Q(x)

// Minimal debugging tool: spawn ONE probe pose, ray-cast a ground-truth
// observation from a separate pose, then print a per-ray breakdown of the
// sensor model's beam-model terms. Run it twice (once with --x/--y/--theta
// at the true pose, once at a suspect wrong pose) and compare -- no
// built-in two-pose comparison, kept out deliberately to stay minimal.
//
// Ray-casting here uses glt.calc_range() directly, per ray -- NOT
// numpy_calc_range_angles() / calc_range_repeat_angles_eval_sensor_model()
// (the real per-particle weighting path's function). Both internally call
// calc_range(y, x, -theta_world + rotation_const - angle) -- an x/y swap
// PLUS a heading rotation (rotation_const=-3pi/2 under our identity world
// transform), baked in unconditionally. Confirmed 2026-08-18 by direct
// test: a pose that's genuinely free space under grid/giant_lut's own
// [col][row] storage, but whose swapped coordinates land on a wall, reads
// ~0 through that function. The x/y swap alone can be cancelled by
// pre-swapping the call's inputs (verified), but the heading rotation
// cannot be cancelled by any single compensating constant without ALSO
// negating the angles array per-ray (worked out 2026-08-18) -- three
// coupled hacks that have to stay correct together, for no actual benefit,
// since calc_range() itself has none of this baggage: giant_lut/grid are
// both built as [x=col][y=row] directly (RangeLib.h's OMap decode loop and
// GiantLUTCast's constructor loop), matching what a human reading (x,y)
// off the map image expects. So: use calc_range() correctly instead of
// compensating for the other function's conventions.
//
// All [setup]-style diagnostics go to stderr; stdout is pure CSV so a
// wrapper script can pipe/parse it directly.

static void print_usage(const char *prog) {
  std::fprintf(
      stderr,
      "Usage: %s --gt-x X --gt-y Y --gt-theta T --x X --y Y --theta T "
      "[--rays N]\n\n"
      "Ray-casts a synthetic ground-truth observation from (--gt-x, --gt-y, "
      "--gt-theta),\n"
      "then scores the probe pose (--x, --y, --theta) against it. By "
      "default uses "
      "glt.calc_range()\n"
      "directly per ray, so --gt-x/--gt-y/--x/--y mean exactly what they "
      "look like on the\n"
      "map (see the 2026-08-18 x/y-swap finding). All poses in map-pixel "
      "coordinates, theta\n"
      "in radians.\n\n"
      "  --rays  optional: number of LIDAR rays, default 60 (matches the "
      "project's usual\n"
      "          downsampled ray count).\n\n"
      "Prints per-ray CSV to stdout: ray,angle,obs_r,pred_d,prob,category,"
      "cum_log_prob\n"
      "category is the beam-model term (hit/short/max/rand) with the "
      "largest raw\n"
      "contribution to that ray's probability, i.e. \"what the sensor model "
      "thinks\n"
      "explains this ray's reading\" -- not a heuristic distance threshold, "
      "the same\n"
      "Z_HIT/Z_SHORT/Z_MAX/Z_RAND formulas build_sensor_model_table() "
      "itself sums.\n",
      prog);
}

// Same 4 terms build_sensor_model_table() sums into sensor_model[r][d], kept
// separate here so the largest one can be reported per-ray instead of only
// their sum. Normalization (build_sensor_model_table's /= norm) scales all 4
// terms by the same per-d constant, so it can't change which one is
// largest -- safe to classify from the raw terms directly.
struct BeamTerms {
  double hit, short_, max_, rand_;
};

static BeamTerms compute_beam_terms(int r, int d, int table_width) {
  BeamTerms t{0.0, 0.0, 0.0, 0.0};
  double z = (double)(r - d);
  t.hit = Z_HIT * std::exp(-(z * z) / (2.0 * SIGMA_HIT * SIGMA_HIT)) /
          (SIGMA_HIT * std::sqrt(2.0 * M_PI));
  if (r < d)
    t.short_ = 2.0 * Z_SHORT * (d - r) / (double)d;
  if (r == table_width - 1)
    t.max_ = Z_MAX;
  if (r < table_width - 1)
    t.rand_ = Z_RAND / (double)(table_width - 1);
  return t;
}

static const char *dominant_category(const BeamTerms &t) {
  double best = t.hit;
  const char *cat = "hit";
  if (t.short_ > best) {
    best = t.short_;
    cat = "short";
  }
  if (t.max_ > best) {
    best = t.max_;
    cat = "max";
  }
  if (t.rand_ > best) {
    best = t.rand_;
    cat = "rand";
  }
  return cat;
}

int main(int argc, char **argv) {
  float gt_x = 0.0f, gt_y = 0.0f, gt_theta = 0.0f;
  float x = 0.0f, y = 0.0f, theta = 0.0f;
  int num_rays = 60;
  bool have_gt = false, have_probe = false;

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
    if (arg == "--gt-x") {
      gt_x = std::atof(value);
      have_gt = true;
    } else if (arg == "--gt-y")
      gt_y = std::atof(value);
    else if (arg == "--gt-theta")
      gt_theta = std::atof(value);
    else if (arg == "--x") {
      x = std::atof(value);
      have_probe = true;
    } else if (arg == "--y")
      y = std::atof(value);
    else if (arg == "--theta")
      theta = std::atof(value);
    else if (arg == "--rays")
      num_rays = std::atoi(value);
    else {
      std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
      print_usage(argv[0]);
      return 1;
    }
  }
  if (!have_gt || !have_probe || num_rays <= 0) {
    print_usage(argv[0]);
    return 1;
  }

  std::fprintf(stderr, "[setup] loading map\n");
  OMap map(QUOTE(MAP_PATH));
  if (map.error()) {
    std::fprintf(stderr, "failed to load map: %s\n", QUOTE(MAP_PATH));
    return 1;
  }
  set_identity_ros_transform(map);

  const int max_range_px = (int)(MAX_RANGE_METERS / MAP_RESOLUTION);
  const int table_width = max_range_px + 1;
  std::fprintf(stderr, "[setup] building sensor model table (width=%d)\n",
               table_width);
  std::vector<double> sensor_table = build_sensor_model_table(table_width);

  std::fprintf(stderr, "[setup] building GiantLUTCast (this is slow)\n");
  GiantLUTCast glt(map, max_range_px, THETA_DISCRETIZATION);
  glt.set_sensor_model(sensor_table.data(), table_width);

  // Catches exactly the confusion this tool hit on 2026-08-18: a pose that
  // LOOKS free in some other PNG/viewer (e.g. maps/basement_fixed.png) can
  // still be occupied in THIS map (maps/basement_fixed_unmapped120.png,
  // MAP_PATH above) -- that mismatch alone explains every ray reading ~0
  // regardless of angle, with no bug in the ray-casting/sensor-model code
  // at all. Printed unconditionally, before any ray-casting, so it's the
  // first thing checked.
  if (map.isOccupied((int)gt_x, (int)gt_y))
    std::fprintf(stderr,
                 "[setup] WARNING: ground-truth pose (%.1f, %.1f) is itself "
                 "occupied in this map -- every obs_r will read ~0\n",
                 gt_x, gt_y);
  if (map.isOccupied((int)x, (int)y))
    std::fprintf(stderr,
                 "[setup] WARNING: probe pose (%.1f, %.1f) is itself "
                 "occupied in this map -- every pred_d will read ~0\n",
                 x, y);

  std::vector<float> angles = make_angles(num_rays);
  std::vector<float> obs(num_rays), pred(num_rays);
  // Deliberately NOT numpy_calc_range_angles() here (2026-08-18) -- that
  // function (and calc_range_repeat_angles_eval_sensor_model, which the real
  // per-particle weighting path uses) internally calls
  // calc_range(y, x, theta - angle), i.e. it swaps x/y before indexing
  // giant_lut/grid, which are both stored [col][row]. Confirmed by direct
  // test (2026-08-18): a pose that's genuinely free space under grid's own
  // [col][row] convention, but whose SWAPPED coordinates land on a wall,
  // reads obs_r~=0 for every ray through numpy_calc_range_angles. calc_range()
  // itself has no such swap -- giant_lut/grid's construction loops in
  // RangeLib.h both index it as [x=col][y=row] directly, matching what a
  // human reading (x,y) off the map image would expect. So this probe calls
  // calc_range() directly, per-ray, so --gt-x/--gt-y/--x/--y mean exactly
  // what they look like on the map (and in --plot). This makes the probe's
  // own ray-casting NOT byte-identical to the real per-particle path
  // anymore -- that's the point, not an oversight: this tool now measures
  // "what does the sensor model see at the pose I actually pointed at,"
  // independent of whichever convention calc_range_repeat_angles_eval_
  // sensor_model turns out to need fixed in the real pipeline.
  // See the long comment above main() for why calc_range() is called
  // directly, per-ray, rather than numpy_calc_range_angles().
  for (int a = 0; a < num_rays; ++a) {
    obs[a] = glt.calc_range(gt_x, gt_y, gt_theta + angles[a]);
    pred[a] = glt.calc_range(x, y, theta + angles[a]);
  }

  std::fprintf(stderr,
               "[setup] gt=(%.1f, %.1f, %.3f) probe=(%.1f, %.1f, %.3f) "
               "rays=%d table_width=%d\n",
               gt_x, gt_y, gt_theta, x, y, theta, num_rays, table_width);

  std::printf("ray,angle,obs_r,pred_d,prob,category,cum_log_prob\n");
  double cum_log_prob = 0.0;
  for (int a = 0; a < num_rays; ++a) {
    float r = obs[a], d = pred[a];
    r = std::min(std::max(r, 0.0f), (float)table_width - 1.0f);
    d = std::min(std::max(d, 0.0f), (float)table_width - 1.0f);
    int ri = (int)r, di = (int)d;
    double prob = sensor_table[ri * table_width + di];
    const char *category = dominant_category(compute_beam_terms(ri, di, table_width));
    cum_log_prob += std::log(prob);
    std::printf("%d,%.6f,%d,%d,%.10e,%s,%.6f\n", a, angles[a], ri, di, prob,
                category, cum_log_prob);
  }

  return 0;
}
