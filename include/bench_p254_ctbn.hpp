#pragma once
#include <CL/sycl.hpp>
#include <ctbignum/ctbignum.hpp>

using namespace cbn::literals;

constexpr auto mod_p254 =
    21888242871839275222246405745257275088548364400416034343698204186575808495617_ZL;
constexpr auto mod_p254_bn = cbn::to_big_int(mod_p254);

using ff_p254_t = decltype(cbn::Zq(mod_p254));

sycl::event benchmark_ff_p254_t_addition(sycl::queue &q, const uint32_t dim,
                                         const uint32_t wg_size,
                                         const uint32_t itr_count);

sycl::event benchmark_ff_p254_t_subtraction(sycl::queue &q, const uint32_t dim,
                                            const uint32_t wg_size,
                                            const uint32_t itr_count);

sycl::event benchmark_ff_p254_t_multiplication(sycl::queue &q,
                                               const uint32_t dim,
                                               const uint32_t wg_size,
                                               const uint32_t itr_count);

sycl::event benchmark_ff_p254_t_division(sycl::queue &q, const uint32_t dim,
                                         const uint32_t wg_size,
                                         const uint32_t itr_count);

sycl::event benchmark_ff_p254_t_inversion(sycl::queue &q, const uint32_t dim,
                                          const uint32_t wg_size,
                                          const uint32_t itr_count);

sycl::event benchmark_ff_p254_t_exponentiation(sycl::queue &q,
                                               const uint32_t dim,
                                               const uint32_t wg_size,
                                               const uint32_t itr_count);
