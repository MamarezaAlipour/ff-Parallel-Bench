#pragma once
#include <CL/sycl.hpp>
#include <ctbignum/ctbignum.hpp>

using namespace cbn::literals;

constexpr auto mod_p64 = 18446744069414584321_ZL;
constexpr auto mod_p64_bn = cbn::to_big_int(mod_p64);

using ff_p64_t = decltype(cbn::Zq(mod_p64));

sycl::event benchmark_ff_p64_t_addition(sycl::queue &q, const uint32_t dim,
                                        const uint32_t wg_size,
                                        const uint32_t itr_count);

sycl::event benchmark_ff_p64_t_subtraction(sycl::queue &q, const uint32_t dim,
                                           const uint32_t wg_size,
                                           const uint32_t itr_count);

sycl::event benchmark_ff_p64_t_multiplication(sycl::queue &q,
                                              const uint32_t dim,
                                              const uint32_t wg_size,
                                              const uint32_t itr_count);

sycl::event benchmark_ff_p64_t_division(sycl::queue &q, const uint32_t dim,
                                        const uint32_t wg_size,
                                        const uint32_t itr_count);

sycl::event benchmark_ff_p64_t_inversion(sycl::queue &q, const uint32_t dim,
                                         const uint32_t wg_size,
                                         const uint32_t itr_count);

sycl::event benchmark_ff_p64_t_exponentiation(sycl::queue &q,
                                              const uint32_t dim,
                                              const uint32_t wg_size,
                                              const uint32_t itr_count);
