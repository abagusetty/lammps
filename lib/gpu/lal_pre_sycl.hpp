//*************************************************************************
//                       Device Configuration Definitions
//                    See lal_preprocessor.h for definitions
//*************************************************************************/

// -------------------------------------------------------------------------
//                           SYCL DEFINITIONS
// -------------------------------------------------------------------------

#if defined(USE_SYCL)

// -------------------------------------------------------------------------
//                             DEVICE CONFIGURATION
// -------------------------------------------------------------------------


#define CONFIG_ID 103
#define SIMD_SIZE 8

#define MEM_THREADS SIMD_SIZE
#define SHUFFLE_AVAIL 1
#define FAST_MATH 1

#define THREADS_PER_ATOM 4
#define THREADS_PER_CHARGE 8
#define THREADS_PER_THREE 2

#define BLOCK_PAIR 256
#define BLOCK_BIO_PAIR 256
#define BLOCK_ELLIPSE 128
#define PPPM_BLOCK_1D 64
#define BLOCK_NBOR_BUILD 128
#define BLOCK_CELL_2D 8
#define BLOCK_CELL_ID 128

#define MAX_SHARED_TYPES 11
#define MAX_BIO_SHARED_TYPES 128
#define PPPM_MAX_SPLINE 8

// -------------------------------------------------------------------------
//                              KERNEL MACROS
// -------------------------------------------------------------------------

#ifdef USE_SYCL
#include <sycl/sycl.hpp>
#endif

#define fast_mul(X,Y) (X)*(Y)

#define EVFLAG 1
#define NOUNROLL
#define GLOBAL_ID_X threadIdx.x+fast_mul(blockIdx.x,blockDim.x)
#define GLOBAL_ID_Y threadIdx.y+fast_mul(blockIdx.y,blockDim.y)
#define GLOBAL_SIZE_X fast_mul(gridDim.x,blockDim.x);
#define GLOBAL_SIZE_Y fast_mul(gridDim.y,blockDim.y);
#define THREAD_ID_X item.get_local_id(2)
#define THREAD_ID_Y item.get_local_id(1)
#define BLOCK_ID_X item.get_group(2)
#define BLOCK_ID_Y item.get_group(1)
#define BLOCK_SIZE_X item.get_local_range().get(2)
#define BLOCK_SIZE_Y item.get_local_range().get(1)
#define NUM_BLOCKS_X gridDim.x

#define atom_add atomicAdd
#define ucl_inline static __inline__ __attribute__((always_inline))

// -------------------------------------------------------------------------
//                         KERNEL MACROS - TEXTURES
// -------------------------------------------------------------------------

// #if defined(__HIP_PLATFORM_HCC__) || defined(__HIP_PLATFORM_AMD__)
// #define _texture(name, type)  __device__ type* name
// #define _texture_2d(name, type)  __device__ type* name
// #else
#define _texture(name, type)  sycl::image<1>* name //texture<type> name
#define _texture_2d(name, type) sycl::image<2>* name //texture<type, cudaTextureType2D> name
// #endif

#if (__CUDACC_VER_MAJOR__ < 11)
  #ifdef _DOUBLE_DOUBLE
  #define fetch4(ans,i,pos_tex) {                        \
    int4 xy = tex1Dfetch(pos_tex,i*2);                   \
    int4 zt = tex1Dfetch(pos_tex,i*2+1);                 \
    ans.x=__hiloint2double(xy.y, xy.x);                  \
    ans.y=__hiloint2double(xy.w, xy.z);                  \
    ans.z=__hiloint2double(zt.y, zt.x);                  \
    ans.w=__hiloint2double(zt.w, zt.z);                  \
  }
  #define fetch(ans,i,q_tex) {                           \
    int2 qt = tex1Dfetch(q_tex,i);                       \
    ans=__hiloint2double(qt.y, qt.x);                    \
  }
  #else
  #define fetch4(ans,i,pos_tex) ans=tex1Dfetch(pos_tex, i);
  #define fetch(ans,i,q_tex) ans=tex1Dfetch(q_tex,i);
  #endif
#else
  #define fetch4(ans,i,x) ans=x[i]
  #define fetch(ans,i,q) ans=q[i]
  #undef _texture
  #undef _texture_2d
  #define _texture(name, type)
  #define _texture_2d(name, type)
  #define pos_tex x_
  #define quat_tex qif
  #define q_tex q_
  #define vel_tex v_
  #define mu_tex mu_
#endif

// -------------------------------------------------------------------------
//                           KERNEL MACROS - MATH
// -------------------------------------------------------------------------

#ifdef _DOUBLE_DOUBLE

#define ucl_exp sycl::exp
#define ucl_powr sycl::powr
#define ucl_atan sycl::atan
#define ucl_cbrt sycl::cbrt
#define ucl_ceil sycl::ceil
#define ucl_abs sycl::fabs
#define ucl_rsqrt sycl::rsqrt
#define ucl_sqrt sycl::sqrt
#define ucl_recip sycl::recip

#else

#define ucl_exp sycl::native::exp
#define ucl_powr sycl::native::powr
#define ucl_atan sycl::atanf
#define ucl_cbrt sycl::cbrtf
#define ucl_ceil sycl::ceilf
#define ucl_abs sycl::fabsf
#define ucl_rsqrt sycl::native::rsqrt
#define ucl_sqrt sycl::native::sqrt
#define ucl_recip sycl::native::recip

#endif

// -------------------------------------------------------------------------
//                         KERNEL MACROS - SHUFFLE
// -------------------------------------------------------------------------

ucl_inline float shfl_down(float var, unsigned int delta, sycl::sub_group& sg) {
  return sg.shuffle_down(var, delta);  
}
ucl_inline float shfl_xor(float var, unsigned int lanemask, sycl::sub_group& sg) {
  return sg.shuffle_xor(var, lanemask);  
}
#define simd_broadcast_i(var, src, width)	\
  __shfl_sync(0xffffffff, var, src, width)
#define simd_broadcast_f(var, src, width)	\
  __shfl_sync(0xffffffff, var, src, width)
#ifdef _DOUBLE_DOUBLE
ucl_inline double simd_broadcast_d(double var, unsigned int src) {
  sycl::group_broadcast();
  int2 tmp;
  tmp.x = __double2hiint(var);
  tmp.y = __double2loint(var);
  tmp.x = __shfl_sync(0xffffffff,tmp.x,src);
  tmp.y = __shfl_sync(0xffffffff,tmp.y,src);
  return __hiloint2double(tmp.x,tmp.y);
}
#endif

// -------------------------------------------------------------------------
//                            END SYCL DEFINITIONS
// -------------------------------------------------------------------------

#endif
