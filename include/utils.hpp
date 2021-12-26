#pragma once
#include <iomanip>
#include <iostream>

void print_benchmark_table_row(const uint64_t dim, const uint64_t itr_cnt,
                               const int64_t total_tm, const double tm_per_op);
