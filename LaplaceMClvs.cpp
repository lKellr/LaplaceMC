#include <fstream>
#include <functional>
#include <iostream>
#include <mdspan>
#include <omp.h>
#include <random>
#include <vector>

constexpr int n_nodes = 200;
constexpr int samples_per_cell = 10;
constexpr int rd_seed = 12345; // std::random_device rd();
constexpr bool post_smooth = false;
constexpr bool use_lvs = true;

const inline std::function<double(double, double)> boundary_func =
    [](double x, double y) {
      return std::sqrt(std::max(0.0, (x + y < 1.)
                                         ? 0.5 - x * x - y * y
                                         : 0.5 - (x - 1.0) * (x - 1.0) -
                                               (y - 1.0) * (y - 1.0)));
    };
const std::function<double(double, double)> boundary_geometry_lvs_func =
    [](double x, double y) {
      return (x + y < 1.) ? x * x + y * y - 0.5
                          : (x - 1.0) * (x - 1.0) + (y - 1.0) * (y - 1.0) - 0.5;
    };
const std::string filename = "results_lvs" + std::to_string(n_nodes) +
                             (post_smooth ? "smooth" : "") + ".csv";

enum smoothing_kernel { FD, MEAN, GAUSS };

void solve_Laplace_random_walk(std::vector<double> &);

void write_array(const std::vector<double>, const int, const std::string);

template <const smoothing_kernel> void smooth(std::vector<double> &, const int);

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;

  // initialize ouput array
  std::vector<double> phi_raw(n_nodes * n_nodes, 0.0);

  solve_Laplace_random_walk(phi_raw);

  // possible smoothing step, problematic w.r.t BCs
  if (post_smooth)
    smooth<FD>(phi_raw, 20);

  // write results
  write_array(phi_raw, n_nodes, filename);

  return 0;
}

void solve_Laplace_random_walk(std::vector<double> &phi_raw) {
  auto phi_md = std::mdspan(phi_raw.data(), n_nodes, n_nodes);

  // evalute level-set field on nodes
  std::vector<double> boundary_geometry_lvs_raw(n_nodes * n_nodes, 0.0);
  auto boundary_geometry_lvs_md =
      std::mdspan(boundary_geometry_lvs_raw.data(), n_nodes, n_nodes);
#pragma omp parallel for collapse(2)
  for (int ix_i = 1; ix_i < n_nodes - 1; ix_i++)
    for (int ix_j = 1; ix_j < n_nodes - 1; ix_j++)
      boundary_geometry_lvs_md[ix_i, ix_j] =
          boundary_geometry_lvs_func(ix_i / static_cast<double>(n_nodes - 1),
                                     ix_j / static_cast<double>(n_nodes - 1));

#pragma omp parallel // num_threads(4)
  {                  // instantiate rng
    std::mt19937 gen(rd_seed + omp_get_thread_num());
    // std::minstd_rand0 gen(rd_seed);
    std::uniform_int_distribution<> distrib(0, 3);

// iterate grid
#pragma omp for collapse(2)
    for (int ix_i = 0; ix_i < n_nodes; ix_i++)
      for (int ix_j = 0; ix_j < n_nodes; ix_j++)
        // iterate samples
        for (int ix_sample = 0; ix_sample < samples_per_cell; ix_sample++) {
          int ix_i_particle = ix_i;
          int ix_j_particle = ix_j;
          // update position
          while (
              ix_i_particle > 0 && ix_i_particle < n_nodes - 1 &&
              ix_j_particle > 0 && ix_j_particle < n_nodes - 1 &&
              (!use_lvs ||
               boundary_geometry_lvs_md[ix_i_particle, ix_j_particle] < 0.0)) {
            const int rnd_dir = distrib(gen);
            if (rnd_dir == 0) {
              ix_i_particle += 1;
              ix_j_particle += 1;
            } else if (rnd_dir == 1) {
              ix_i_particle -= 1;
              ix_j_particle += 1;
            } else if (rnd_dir == 2) {
              ix_i_particle += 1;
              ix_j_particle -= 1;
            } else {
              ix_i_particle -= 1;
              ix_j_particle -= 1;
            }
          }
          // update expectation value
          phi_md[ix_i, ix_j] +=
              (boundary_func(ix_i_particle / static_cast<double>(n_nodes - 1),
                             ix_j_particle / static_cast<double>(n_nodes - 1)) -
               phi_md[ix_i, ix_j]) /
              (ix_sample + 1);
        }
  }
}

void write_array(const std::vector<double> array, const int array_size,
                 const std::string filename) {
  std::cout << "outputting results as " << filename << std::endl;
  std::ofstream output_file;
  output_file.open(filename, std::ios::trunc);

  const auto array_view_md = std::mdspan(array.data(), array_size, array_size);

  for (int ix_i = 0; ix_i < array_size; ix_i++) {
    bool first = true;
    for (int ix_j = 0; ix_j < array_size; ix_j++) {
      if (!first) {
        output_file << ",";
      }
      first = false;
      output_file << array_view_md[ix_i, ix_j];
    }
    output_file << "\n";
  }
  output_file.close();
}

template <const smoothing_kernel filter_type>
void smooth(std::vector<double> &array, const int iterations) {

  std::vector<double> array_tmp_raw(n_nodes * n_nodes, 0.0);
  auto array_tmp_md = std::mdspan(array_tmp_raw.data(), n_nodes, n_nodes);
  auto array_in_md = std::mdspan(array.data(), n_nodes, n_nodes);

  // fill boundary values
  for (int ix = 0; ix < n_nodes; ix++) {
    array_tmp_md[0, ix] = array_in_md[0, ix];
    array_tmp_md[n_nodes - 1, ix] = array_in_md[n_nodes - 1, ix];
    array_tmp_md[ix, 0] = array_in_md[ix, 0];
    array_tmp_md[ix, n_nodes - 1] = array_in_md[ix, n_nodes - 1];
  }

#pragma omp parallel
  for (int ix_iter = 0; ix_iter < iterations; ix_iter++) {
#pragma omp for collapse(2)
    for (int ix_i = 1; ix_i < n_nodes - 1; ix_i++)
      for (int ix_j = 1; ix_j < n_nodes - 1; ix_j++)
        switch (filter_type) {
        case FD: {
          constexpr double cfl = 0.2;
          array_tmp_md[ix_i, ix_j] =
              array_in_md[ix_i, ix_j] * (1. - 4. * cfl) +
              cfl * (array_in_md[ix_i + 1, ix_j] + array_in_md[ix_i - 1, ix_j] +
                     array_in_md[ix_i, ix_j + 1] + array_in_md[ix_i, ix_j - 1]);
          break;
        }
        case MEAN: {
          array_tmp_md[ix_i, ix_j] =
              1. / 9. *
              (array_in_md[ix_i, ix_j] + array_in_md[ix_i + 1, ix_j] +
               array_in_md[ix_i + 1, ix_j + 1] + array_in_md[ix_i, ix_j + 1] +
               array_in_md[ix_i - 1, ix_j + 1] + array_in_md[ix_i - 1, ix_j] +
               array_in_md[ix_i - 1, ix_j - 1] + array_in_md[ix_i, ix_j - 1] +
               array_in_md[ix_i + 1, ix_j - 1]);
          break;
        }
        case GAUSS: {
          array_tmp_md[ix_i, ix_j] =
              1. / 16. *
              (4 * array_in_md[ix_i, ix_j] + 2 * array_in_md[ix_i + 1, ix_j] +
               array_in_md[ix_i + 1, ix_j + 1] +
               2 * array_in_md[ix_i, ix_j + 1] +
               array_in_md[ix_i - 1, ix_j + 1] +
               2 * array_in_md[ix_i - 1, ix_j] +
               array_in_md[ix_i - 1, ix_j - 1] +
               2 * array_in_md[ix_i, ix_j - 1] +
               array_in_md[ix_i + 1, ix_j - 1]);
          break;
        }
        }
#pragma omp single
    std::swap(array_tmp_md, array_in_md);
  }
}
