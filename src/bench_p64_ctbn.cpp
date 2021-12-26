#include <bench_p64_ctbn.hpp>

sycl::event benchmark_ff_p64_t_addition(sycl::queue &q, const uint32_t dim,
                                        const uint32_t wg_size,
                                        const uint32_t itr_count) {
  return q.submit([&](sycl::handler &h) {
    // allocate some space in local memory where I'll write
    // some garbage data, at end of computation so that compiler
    // doesn't end up optimizing too much !
    sycl::accessor<ff_p64_t, 1, sycl::access_mode::read_write,
                   sycl::target::local>
        lds{sycl::range<1>{wg_size}, h};

    h.parallel_for<class kernelFF_p64_Addition>(
        sycl::nd_range<2>{sycl::range<2>{dim, dim}, sycl::range<2>{1, wg_size}},
        [=](sycl::nd_item<2> it) {
          const size_t loc_lid = it.get_local_linear_id();

          ff_p64_t op(2147483648_ZL);
          ff_p64_t tmp(576460752303423488_ZL);

          for (uint64_t i = 0ul; i < itr_count; i++) {
            tmp += op;
          }

          // every work item writes back to local memory
          lds[loc_lid] = tmp;
        });
  });
}

sycl::event benchmark_ff_p64_t_subtraction(sycl::queue &q, const uint32_t dim,
                                           const uint32_t wg_size,
                                           const uint32_t itr_count) {
  return q.submit([&](sycl::handler &h) {
    // allocate some space in local memory where I'll write
    // some garbage data, at end of computation so that compiler
    // doesn't end up optimizing too much !
    sycl::accessor<ff_p64_t, 1, sycl::access_mode::read_write,
                   sycl::target::local>
        lds{sycl::range<1>{wg_size}, h};

    h.parallel_for<class kernelFF_p64_Subtraction>(
        sycl::nd_range<2>{sycl::range<2>{dim, dim}, sycl::range<2>{1, wg_size}},
        [=](sycl::nd_item<2> it) {
          const size_t loc_lid = it.get_local_linear_id();

          ff_p64_t op(2147483648_ZL);
          ff_p64_t tmp(576460752303423488_ZL);

          for (uint64_t i = 0ul; i < itr_count; i++) {
            tmp -= op;
          }

          // every work item writes back to local memory
          lds[loc_lid] = tmp;
        });
  });
}

sycl::event benchmark_ff_p64_t_multiplication(sycl::queue &q,
                                              const uint32_t dim,
                                              const uint32_t wg_size,
                                              const uint32_t itr_count) {
  return q.submit([&](sycl::handler &h) {
    // allocate some space in local memory where I'll write
    // some garbage data, at end of computation so that compiler
    // doesn't end up optimizing too much !
    sycl::accessor<ff_p64_t, 1, sycl::access_mode::read_write,
                   sycl::target::local>
        lds{sycl::range<1>{wg_size}, h};

    h.parallel_for<class kernelFF_p64_Multiplication>(
        sycl::nd_range<2>{sycl::range<2>{dim, dim}, sycl::range<2>{1, wg_size}},
        [=](sycl::nd_item<2> it) {
          const size_t loc_lid = it.get_local_linear_id();

          ff_p64_t op(2147483648_ZL);
          ff_p64_t tmp(576460752303423488_ZL);

          for (uint64_t i = 0ul; i < itr_count; i++) {
            tmp *= op;
          }

          // every work item writes back to local memory
          lds[loc_lid] = tmp;
        });
  });
}

sycl::event benchmark_ff_p64_t_division(sycl::queue &q, const uint32_t dim,
                                        const uint32_t wg_size,
                                        const uint32_t itr_count) {
  return q.submit([&](sycl::handler &h) {
    // allocate some space in local memory where I'll write
    // some garbage data, at end of computation so that compiler
    // doesn't end up optimizing too much !
    sycl::accessor<ff_p64_t, 1, sycl::access_mode::read_write,
                   sycl::target::local>
        lds{sycl::range<1>{wg_size}, h};

    h.parallel_for<class kernelFF_p64_Division>(
        sycl::nd_range<2>{sycl::range<2>{dim, dim}, sycl::range<2>{1, wg_size}},
        [=](sycl::nd_item<2> it) {
          const size_t loc_lid = it.get_local_linear_id();

          ff_p64_t op(2147483648_ZL);
          ff_p64_t tmp(576460752303423488_ZL);

          for (uint64_t i = 0ul; i < itr_count; i++) {
            tmp /= op;
          }

          // every work item writes back to local memory
          lds[loc_lid] = tmp;
        });
  });
}

sycl::event benchmark_ff_p64_t_inversion(sycl::queue &q, const uint32_t dim,
                                         const uint32_t wg_size,
                                         const uint32_t itr_count) {
  return q.submit([&](sycl::handler &h) {
    // allocate some space in local memory where I'll write
    // some garbage data, at end of computation so that compiler
    // doesn't end up optimizing too much !
    sycl::accessor<ff_p64_t, 1, sycl::access_mode::read_write,
                   sycl::target::local>
        lds{sycl::range<1>{wg_size}, h};

    h.parallel_for<class kernelFF_p64_Inversion>(
        sycl::nd_range<2>{sycl::range<2>{dim, dim}, sycl::range<2>{1, wg_size}},
        [=](sycl::nd_item<2> it) {
          const size_t loc_lid = it.get_local_linear_id();

          ff_p64_t tmp(576460752303423488_ZL);

          for (uint64_t i = 0ul; i < itr_count; i++) {
            tmp = static_cast<ff_p64_t>(cbn::mod_inv(tmp.data, mod_p64_bn));
          }

          // every work item writes back to local memory
          lds[loc_lid] = tmp;
        });
  });
}

sycl::event benchmark_ff_p64_t_exponentiation(sycl::queue &q,
                                              const uint32_t dim,
                                              const uint32_t wg_size,
                                              const uint32_t itr_count) {
  return q.submit([&](sycl::handler &h) {
    // allocate some space in local memory where I'll write
    // some garbage data, at end of computation so that compiler
    // doesn't end up optimizing too much !
    sycl::accessor<ff_p64_t, 1, sycl::access_mode::read_write,
                   sycl::target::local>
        lds{sycl::range<1>{wg_size}, h};

    h.parallel_for<class kernelFF_p64_Exponentiation>(
        sycl::nd_range<2>{sycl::range<2>{dim, dim}, sycl::range<2>{1, wg_size}},
        [=](sycl::nd_item<2> it) {
          const size_t loc_lid = it.get_local_linear_id();

          ff_p64_t op(2147483648_ZL);
          ff_p64_t tmp(576460752303423488_ZL);

          for (uint64_t i = 0ul; i < itr_count; i++) {
            tmp = static_cast<ff_p64_t>(
                cbn::mod_exp(op.data, tmp.data, mod_p64_bn));
          }

          // every work item writes back to local memory
          lds[loc_lid] = tmp;
        });
  });
}
