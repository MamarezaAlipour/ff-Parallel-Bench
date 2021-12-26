#include <utils.hpp>

void print_benchmark_table_row(const uint64_t dim, const uint64_t itr_cnt,
                               const int64_t total_tm, const double tm_per_op) {
  std::cout << std::setw(5) << std::left << dim << "x" << std::setw(5)
            << std::right << dim << "\t\t" << std::setw(8) << std::right
            << itr_cnt << "\t\t" << std::setw(15) << std::right << total_tm
            << " ns"
            << "\t\t" << std::setw(15) << std::right << tm_per_op << " ns"
            << "\t\t" << std::setw(22) << std::right << 1e9 / tm_per_op
            << std::endl;
}
