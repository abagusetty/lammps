#ifndef SYCL_MACROS_HPP
#define SYCL_MACROS_HPP

#include <cstdio>
#include <cassert>

#include <sycl/sycl.hpp>
#include <level_zero/ze_api.h>

#ifdef MPI_GERYON
#include "mpi.h"
#define SYCL_GERYON_EXIT do {                   \
    int is_final;                               \
    MPI_Finalized(&is_final);                   \
    if (!is_final)                              \
      MPI_Abort(MPI_COMM_WORLD,-1);             \
  } while(0)
#else
#define SYCL_GERYON_EXIT assert(0==1)
#endif

#ifndef UCL_GERYON_EXIT
#define UCL_GERYON_EXIT SYCL_GERYON_EXIT
#endif

#  define ZE_SAFE_CALL( call) do {                                      \
    ze_result_t err = call;                                             \
    if(err != ZE_RESULT_SUCCESS) {					\
      fprintf(stderr, "SYCL (L0_backend) error in file '%s' in line %i : %d.\n", \
              __FILE__, __LINE__, err );                                \
      SYCL_GERYON_EXIT;                                                 \
    } } while (0)

#endif // SYCL_MACROS_HPP
