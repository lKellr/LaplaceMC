#include <fstream>
#include <functional>
#include <iostream>
#include <mdspan>
#include <omp.h>
#include <random>
#include <vector>

constexpr int n_nodes = 500;
constexpr int samples_per_cell = 10;
constexpr int rd_seed = 12345; // std::random_device rd();
constexpr bool post_smooth = false;
const inline std::function<double(double, double)> boundary_func =
    [](double x, double y) { return x + y - 1.; };
const std::function<double(double, double)> boundary_geometry_lvs;
const std::string filename = "results" + std::to_string(n_nodes) +
                             (post_smooth ? "smooth" : "") + ".csv";

void write_array(const std::vector<double>, const int, const std::string);

template <const int>
std::vector<double> smooth(const std::vector<double> &, const int,
                           const int = 1);

int main(int argc, char **argv) {

  // initialize ouput array

  std::vector<double> phi_raw(n_nodes * n_nodes, 0.0);
  auto phi_md = std::mdspan(phi_raw.data(), n_nodes, n_nodes);

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
          while (ix_i_particle > 0 && ix_i_particle < n_nodes - 1 &&
                 ix_j_particle > 0 && ix_j_particle < n_nodes - 1) {
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

  if (post_smooth)
    phi_raw = smooth<1>(phi_raw, n_nodes, 5);

  // write results
  write_array(phi_raw, n_nodes, filename);

  return 0;
}

void write_array(const std::vector<double> array, const int array_size,
                 const std::string filename) {
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

template <const int filter_type>
std::vector<double> smooth(const std::vector<double> &array,
                           const int array_size, const int iterations) {

  std::vector<double> array_out_raw(array_size * array_size, 0.0);
  const auto array_out_md =
      std::mdspan(array_out_raw.data(), array_size, array_size);
  const auto array_in_md = std::mdspan(array.data(), array_size, array_size);

  // fill boundary values
  for (int ix = 0; ix < n_nodes; ix++) {
    array_out_md[0, ix] = array_in_md[0, ix];
    array_out_md[n_nodes - 1, ix] = array_in_md[n_nodes - 1, ix];
    array_out_md[ix, 0] = array_in_md[ix, 0];
    array_out_md[ix, n_nodes - 1] = array_in_md[ix, n_nodes - 1];
  }

#pragma omp parallel
  for (int ix_iter = 0; ix_iter < iterations; ix_iter++) {
#pragma omp for collapse(2)
    for (int ix_i = 1; ix_i < n_nodes - 1; ix_i++)
      for (int ix_j = 1; ix_j < n_nodes - 1; ix_j++)
        switch (filter_type) {
        case 0:
          array_out_md[ix_i, ix_j] =
              0.25 *
              (array_in_md[ix_i + 1, ix_j] + array_in_md[ix_i - 1, ix_j] +
               array_in_md[ix_i, ix_j + 1] + array_in_md[ix_i, ix_j - 1]);
          break;
        case 1:
          array_out_md[ix_i, ix_j] =
              1. / 9. *
              (array_in_md[ix_i, ix_j] + array_in_md[ix_i + 1, ix_j] +
               array_in_md[ix_i + 1, ix_j + 1] + array_in_md[ix_i, ix_j + 1] +
               array_in_md[ix_i - 1, ix_j + 1] + array_in_md[ix_i - 1, ix_j] +
               array_in_md[ix_i - 1, ix_j - 1] + array_in_md[ix_i, ix_j - 1] +
               array_in_md[ix_i + 1, ix_j - 1]);
          break;
        case 2:
          array_out_md[ix_i, ix_j] =
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
#pragma omp barrier
  }
  return array_out_raw;
}
