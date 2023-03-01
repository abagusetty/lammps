// clang-format off
/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifndef LMP_LMPTYPE_SYCL_H
#define LMP_LMPTYPE_SYCL_H

#include "pointers.h"
#include "lmptype.h"

#include <sycl/sycl.hpp>
#include <experimental/mdspan>
namespace stdex = std::experimental;

constexpr int FULL = 1;
constexpr int HALFTHREAD = 2;
constexpr int HALF = 4;

#if defined(SYCL_ENABLE_CXX11)
#undef ISFINITE
#define ISFINITE(x) sycl::isfinite(x)
#endif

#if defined(SYCL_ENABLE_CUDA) || defined(SYCL_ENABLE_HIP) || defined(SYCL_ENABLE_INTEL)
#define LMP_SYCL_GPU
#endif

#if defined(LMP_SYCL_GPU)
#define SYCL_GPU_ARG(x) x
#else
#define SYCL_GPU_ARG(x)
#endif

#define MAX_TYPES_STACKPARAMS 12
static constexpr LAMMPS_NS::bigint LMP_SYCL_AV_DELTA = 10;

template<class Scalar_type>
using t_scalar3 = sycl::vec<Scalar_type, 3>;

// Determine memory traits for force array
// Do atomic trait when running HALFTHREAD neighbor list style
template<int NEIGHFLAG>
struct AtomicF {
  enum {value = Kokkos::Unmanaged};
};

template<>
struct AtomicF<HALFTHREAD> {
  enum {value = Kokkos::Atomic|Kokkos::Unmanaged};
};


// Determine memory traits for force array
// Do atomic trait when running HALFTHREAD neighbor list style with CUDA
template<int NEIGHFLAG, class DeviceType>
struct AtomicDup {
  using value = Kokkos::Experimental::ScatterNonAtomic;
};

#ifdef SYCL_ENABLE_CUDA
template<>
struct AtomicDup<HALFTHREAD,Kokkos::Cuda> {
  using value = Kokkos::Experimental::ScatterAtomic;
};
#elif defined(SYCL_ENABLE_HIP)
template<>
struct AtomicDup<HALFTHREAD,Kokkos::Experimental::HIP> {
  using value = Kokkos::Experimental::ScatterAtomic;
};
#elif defined(SYCL_ENABLE_INTEL)
template<>
struct AtomicDup<HALFTHREAD,Kokkos::Experimental::SYCL> {
  using value = Kokkos::Experimental::ScatterAtomic;
};
#endif

#ifdef LMP_SYCL_USE_ATOMICS

#ifdef SYCL_ENABLE_THREADS
template<>
struct AtomicDup<HALFTHREAD,Kokkos::Threads> {
  using value = Kokkos::Experimental::ScatterAtomic;
};
#endif

#endif


// Determine duplication traits for force array
// Use duplication when running threaded and not using atomics
template<int NEIGHFLAG, class DeviceType>
struct NeedDup {
  using value = Kokkos::Experimental::ScatterNonDuplicated;
};

#ifndef LMP_SYCL_USE_ATOMICS

#ifdef SYCL_ENABLE_OPENMP
template<>
struct NeedDup<HALFTHREAD,Kokkos::OpenMP> {
  using value = Kokkos::Experimental::ScatterDuplicated;
};
#endif

#ifdef SYCL_ENABLE_THREADS
template<>
struct NeedDup<HALFTHREAD,Kokkos::Threads> {
  using value = Kokkos::Experimental::ScatterDuplicated;
};
#endif

#endif

template<typename value, typename T1, typename T2>
class ScatterViewHelper {};

template<typename T1, typename T2>
class ScatterViewHelper<Kokkos::Experimental::ScatterDuplicated,T1,T2> {
public:
  __attribute__((always_inline))
  static T1 get(const T1 &dup, const T2 & /*nondup*/) {
    return dup;
  }
};

template<typename T1, typename T2>
class ScatterViewHelper<Kokkos::Experimental::ScatterNonDuplicated,T1,T2> {
public:
  __attribute__((always_inline))
  static T2 get(const T1 & /*dup*/, const T2 &nondup) {
    return nondup;
  }
};


// define precision
// handle global precision, force, energy, positions, kspace separately

#ifndef PRECISION
#define PRECISION 2
#endif
#if PRECISION==1
typedef float LMP_FLOAT;
#else
typedef double LMP_FLOAT;
#endif

#ifndef PREC_FORCE
#define PREC_FORCE PRECISION
#endif

#if PREC_FORCE==1
typedef float F_FLOAT;
#else
typedef double F_FLOAT;
#endif

#ifndef PREC_ENERGY
#define PREC_ENERGY PRECISION
#endif

#if PREC_ENERGY==1
typedef float E_FLOAT;
#else
typedef double E_FLOAT;
#endif

#ifndef PREC_POS
#define PREC_POS PRECISION
#endif

#if PREC_POS==1
typedef float X_FLOAT;
#else
typedef double X_FLOAT;
#endif

#ifndef PREC_VELOCITIES
#define PREC_VELOCITIES PRECISION
#endif

#if PREC_VELOCITIES==1
typedef float V_FLOAT;
#else
typedef double V_FLOAT;
#endif

#if PREC_KSPACE==1
typedef float K_FLOAT;
#else
typedef double K_FLOAT;
#endif

typedef int T_INT;

// ------------------------------------------------------------------------

// LAMMPS types

typedef Kokkos::UnorderedMap<LAMMPS_NS::tagint,int,LMPDeviceType> hash_type;
typedef hash_type::HostMirror host_hash_type;

struct dual_hash_type {
  hash_type d_view;
  host_hash_type h_view;

  template<class DeviceType>
  __attribute__((always_inline))
  std::enable_if_t<(std::is_same_v<DeviceType,LMPDeviceType> || Kokkos::SpaceAccessibility<LMPDeviceType::memory_space,LMPHostType::memory_space>::accessible),hash_type&> view() {return d_view;}

  template<class DeviceType>
  __attribute__((always_inline))
  std::enable_if_t<!(std::is_same_v<DeviceType,LMPDeviceType> || Kokkos::SpaceAccessibility<LMPDeviceType::memory_space,LMPHostType::memory_space>::accessible),host_hash_type&> view() {return h_view;}

};

struct ArrayTypes;

struct ArrayTypes {

// scalar types

typedef Kokkos::DualView<int, LMPDeviceType::array_layout, LMPDeviceType> tdual_int_scalar;
typedef tdual_int_scalar::t_dev t_int_scalar;
typedef tdual_int_scalar::t_dev_const t_int_scalar_const;
typedef tdual_int_scalar::t_dev_um t_int_scalar_um;
typedef tdual_int_scalar::t_dev_const_um t_int_scalar_const_um;

typedef Kokkos::DualView<LMP_FLOAT, LMPDeviceType::array_layout, LMPDeviceType> tdual_float_scalar;
typedef tdual_float_scalar::t_dev t_float_scalar;
typedef tdual_float_scalar::t_dev_const t_float_scalar_const;
typedef tdual_float_scalar::t_dev_um t_float_scalar_um;
typedef tdual_float_scalar::t_dev_const_um t_float_scalar_const_um;

// generic array types

typedef Kokkos::DualView<int*, LMPDeviceType::array_layout, LMPDeviceType> tdual_int_1d;
typedef tdual_int_1d::t_dev t_int_1d;
typedef tdual_int_1d::t_dev_const t_int_1d_const;
typedef tdual_int_1d::t_dev_um t_int_1d_um;
typedef tdual_int_1d::t_dev_const_um t_int_1d_const_um;
typedef tdual_int_1d::t_dev_const_randomread t_int_1d_randomread;

typedef Kokkos::DualView<int*[3], Kokkos::LayoutRight, LMPDeviceType> tdual_int_1d_3;
typedef tdual_int_1d_3::t_dev t_int_1d_3;
typedef tdual_int_1d_3::t_dev_const t_int_1d_3_const;
typedef tdual_int_1d_3::t_dev_um t_int_1d_3_um;
typedef tdual_int_1d_3::t_dev_const_um t_int_1d_3_const_um;
typedef tdual_int_1d_3::t_dev_const_randomread t_int_1d_3_randomread;

typedef Kokkos::DualView<int**, Kokkos::LayoutRight, LMPDeviceType> tdual_int_2d;
typedef tdual_int_2d::t_dev t_int_2d;
typedef tdual_int_2d::t_dev_const t_int_2d_const;
typedef tdual_int_2d::t_dev_um t_int_2d_um;
typedef tdual_int_2d::t_dev_const_um t_int_2d_const_um;
typedef tdual_int_2d::t_dev_const_randomread t_int_2d_randomread;

typedef Kokkos::DualView<int**, LMPDeviceType::array_layout, LMPDeviceType> tdual_int_2d_dl;
typedef tdual_int_2d_dl::t_dev t_int_2d_dl;
typedef tdual_int_2d_dl::t_dev_const t_int_2d_const_dl;
typedef tdual_int_2d_dl::t_dev_um t_int_2d_um_dl;
typedef tdual_int_2d_dl::t_dev_const_um t_int_2d_const_um_dl;
typedef tdual_int_2d_dl::t_dev_const_randomread t_int_2d_randomread_dl;

typedef Kokkos::ualView<LAMMPS_NS::tagint*, LMPDeviceType::array_layout, LMPDeviceType> tdual_tagint_1d;
typedef tdual_tagint_1d::t_dev t_tagint_1d;
typedef tdual_tagint_1d::t_dev_const t_tagint_1d_const;
typedef tdual_tagint_1d::t_dev_um t_tagint_1d_um;
typedef tdual_tagint_1d::t_dev_const_um t_tagint_1d_const_um;
typedef tdual_tagint_1d::t_dev_const_randomread t_tagint_1d_randomread;

typedef Kokkos::DualView<LAMMPS_NS::tagint**, Kokkos::LayoutRight, LMPDeviceType> tdual_tagint_2d;
typedef tdual_tagint_2d::t_dev t_tagint_2d;
typedef tdual_tagint_2d::t_dev_const t_tagint_2d_const;
typedef tdual_tagint_2d::t_dev_um t_tagint_2d_um;
typedef tdual_tagint_2d::t_dev_const_um t_tagint_2d_const_um;
typedef tdual_tagint_2d::t_dev_const_randomread t_tagint_2d_randomread;

typedef Kokkos::DualView<LAMMPS_NS::imageint*, LMPDeviceType::array_layout, LMPDeviceType> tdual_imageint_1d;
typedef tdual_imageint_1d::t_dev t_imageint_1d;
typedef tdual_imageint_1d::t_dev_const t_imageint_1d_const;
typedef tdual_imageint_1d::t_dev_um t_imageint_1d_um;
typedef tdual_imageint_1d::t_dev_const_um t_imageint_1d_const_um;
typedef tdual_imageint_1d::t_dev_const_randomread t_imageint_1d_randomread;

typedef Kokkos::DualView<double*, Kokkos::LayoutRight, LMPDeviceType> tdual_double_1d;
typedef tdual_double_1d::t_dev t_double_1d;
typedef tdual_double_1d::t_dev_const t_double_1d_const;
typedef tdual_double_1d::t_dev_um t_double_1d_um;
typedef tdual_double_1d::t_dev_const_um t_double_1d_const_um;
typedef tdual_double_1d::t_dev_const_randomread t_double_1d_randomread;

typedef Kokkos::DualView<double**, Kokkos::LayoutRight, LMPDeviceType> tdual_double_2d;
typedef tdual_double_2d::t_dev t_double_2d;
typedef tdual_double_2d::t_dev_const t_double_2d_const;
typedef tdual_double_2d::t_dev_um t_double_2d_um;
typedef tdual_double_2d::t_dev_const_um t_double_2d_const_um;
typedef tdual_double_2d::t_dev_const_randomread t_double_2d_randomread;

// 1d float array n

typedef Kokkos::DualView<LMP_FLOAT*, LMPDeviceType::array_layout, LMPDeviceType> tdual_float_1d;
typedef tdual_float_1d::t_dev t_float_1d;
typedef tdual_float_1d::t_dev_const t_float_1d_const;
typedef tdual_float_1d::t_dev_um t_float_1d_um;
typedef tdual_float_1d::t_dev_const_um t_float_1d_const_um;
typedef tdual_float_1d::t_dev_const_randomread t_float_1d_randomread;

//2d float array n
typedef Kokkos::DualView<LMP_FLOAT**, Kokkos::LayoutRight, LMPDeviceType> tdual_float_2d;
typedef tdual_float_2d::t_dev t_float_2d;
typedef tdual_float_2d::t_dev_const t_float_2d_const;
typedef tdual_float_2d::t_dev_um t_float_2d_um;
typedef tdual_float_2d::t_dev_const_um t_float_2d_const_um;
typedef tdual_float_2d::t_dev_const_randomread t_float_2d_randomread;

//Position Types
//1d X_FLOAT array n
typedef Kokkos::DualView<X_FLOAT*, LMPDeviceType::array_layout, LMPDeviceType> tdual_xfloat_1d;
typedef tdual_xfloat_1d::t_dev t_xfloat_1d;
typedef tdual_xfloat_1d::t_dev_const t_xfloat_1d_const;
typedef tdual_xfloat_1d::t_dev_um t_xfloat_1d_um;
typedef tdual_xfloat_1d::t_dev_const_um t_xfloat_1d_const_um;
typedef tdual_xfloat_1d::t_dev_const_randomread t_xfloat_1d_randomread;

//2d X_FLOAT array n*m
typedef Kokkos::DualView<X_FLOAT**, Kokkos::LayoutRight, LMPDeviceType> tdual_xfloat_2d;
typedef tdual_xfloat_2d::t_dev t_xfloat_2d;
typedef tdual_xfloat_2d::t_dev_const t_xfloat_2d_const;
typedef tdual_xfloat_2d::t_dev_um t_xfloat_2d_um;
typedef tdual_xfloat_2d::t_dev_const_um t_xfloat_2d_const_um;
typedef tdual_xfloat_2d::t_dev_const_randomread t_xfloat_2d_randomread;

//2d X_FLOAT array n*4
#ifdef LMP_SYCL_NO_LEGACY
typedef Kokkos::DualView<X_FLOAT*[3], Kokkos::LayoutLeft, LMPDeviceType> tdual_x_array;
#else
typedef Kokkos::DualView<X_FLOAT*[3], Kokkos::LayoutRight, LMPDeviceType> tdual_x_array;
#endif
typedef tdual_x_array::t_dev t_x_array;
typedef tdual_x_array::t_dev_const t_x_array_const;
typedef tdual_x_array::t_dev_um t_x_array_um;
typedef tdual_x_array::t_dev_const_um t_x_array_const_um;
typedef tdual_x_array::t_dev_const_randomread t_x_array_randomread;

//Velocity Types
//1d V_FLOAT array n
typedef Kokkos::DualView<V_FLOAT*, LMPDeviceType::array_layout, LMPDeviceType> tdual_vfloat_1d;
typedef tdual_vfloat_1d::t_dev t_vfloat_1d;
typedef tdual_vfloat_1d::t_dev_const t_vfloat_1d_const;
typedef tdual_vfloat_1d::t_dev_um t_vfloat_1d_um;
typedef tdual_vfloat_1d::t_dev_const_um t_vfloat_1d_const_um;
typedef tdual_vfloat_1d::t_dev_const_randomread t_vfloat_1d_randomread;

//2d V_FLOAT array n*m
typedef Kokkos::DualView<V_FLOAT**, Kokkos::LayoutRight, LMPDeviceType> tdual_vfloat_2d;
typedef tdual_vfloat_2d::t_dev t_vfloat_2d;
typedef tdual_vfloat_2d::t_dev_const t_vfloat_2d_const;
typedef tdual_vfloat_2d::t_dev_um t_vfloat_2d_um;
typedef tdual_vfloat_2d::t_dev_const_um t_vfloat_2d_const_um;
typedef tdual_vfloat_2d::t_dev_const_randomread t_vfloat_2d_randomread;

//2d V_FLOAT array n*3
typedef Kokkos::DualView<V_FLOAT*[3], Kokkos::LayoutRight, LMPDeviceType> tdual_v_array;
//typedef Kokkos::DualView<V_FLOAT*[3], LMPDeviceType::array_layout, LMPDeviceType> tdual_v_array;
typedef tdual_v_array::t_dev t_v_array;
typedef tdual_v_array::t_dev_const t_v_array_const;
typedef tdual_v_array::t_dev_um t_v_array_um;
typedef tdual_v_array::t_dev_const_um t_v_array_const_um;
typedef tdual_v_array::t_dev_const_randomread t_v_array_randomread;

//Force Types
//1d F_FLOAT array n

typedef Kokkos::DualView<F_FLOAT*, LMPDeviceType::array_layout, LMPDeviceType> tdual_ffloat_1d;
typedef tdual_ffloat_1d::t_dev t_ffloat_1d;
typedef tdual_ffloat_1d::t_dev_const t_ffloat_1d_const;
typedef tdual_ffloat_1d::t_dev_um t_ffloat_1d_um;
typedef tdual_ffloat_1d::t_dev_const_um t_ffloat_1d_const_um;
typedef tdual_ffloat_1d::t_dev_const_randomread t_ffloat_1d_randomread;

//2d F_FLOAT array n*m

typedef Kokkos::DualView<F_FLOAT**, Kokkos::LayoutRight, LMPDeviceType> tdual_ffloat_2d;
typedef tdual_ffloat_2d::t_dev t_ffloat_2d;
typedef tdual_ffloat_2d::t_dev_const t_ffloat_2d_const;
typedef tdual_ffloat_2d::t_dev_um t_ffloat_2d_um;
typedef tdual_ffloat_2d::t_dev_const_um t_ffloat_2d_const_um;
typedef tdual_ffloat_2d::t_dev_const_randomread t_ffloat_2d_randomread;

//2d F_FLOAT array n*m, device layout

typedef Kokkos::DualView<F_FLOAT**, LMPDeviceType::array_layout, LMPDeviceType> tdual_ffloat_2d_dl;
typedef tdual_ffloat_2d_dl::t_dev t_ffloat_2d_dl;
typedef tdual_ffloat_2d_dl::t_dev_const t_ffloat_2d_const_dl;
typedef tdual_ffloat_2d_dl::t_dev_um t_ffloat_2d_um_dl;
typedef tdual_ffloat_2d_dl::t_dev_const_um t_ffloat_2d_const_um_dl;
typedef tdual_ffloat_2d_dl::t_dev_const_randomread t_ffloat_2d_randomread_dl;

//2d F_FLOAT array n*3

typedef Kokkos::DualView<F_FLOAT*[3], Kokkos::LayoutRight, LMPDeviceType> tdual_f_array;
//typedef Kokkos::DualView<F_FLOAT*[3], LMPDeviceType::array_layout, LMPDeviceType> tdual_f_array;
typedef tdual_f_array::t_dev t_f_array;
typedef tdual_f_array::t_dev_const t_f_array_const;
typedef tdual_f_array::t_dev_um t_f_array_um;
typedef tdual_f_array::t_dev_const_um t_f_array_const_um;
typedef tdual_f_array::t_dev_const_randomread t_f_array_randomread;

//2d F_FLOAT array n*6 (for virial)

typedef Kokkos::DualView<F_FLOAT*[6], Kokkos::LayoutRight, LMPDeviceType> tdual_virial_array;
typedef tdual_virial_array::t_dev t_virial_array;
typedef tdual_virial_array::t_dev_const t_virial_array_const;
typedef tdual_virial_array::t_dev_um t_virial_array_um;
typedef tdual_virial_array::t_dev_const_um t_virial_array_const_um;
typedef tdual_virial_array::t_dev_const_randomread t_virial_array_randomread;

//Energy Types
//1d E_FLOAT array n

typedef Kokkos::DualView<E_FLOAT*, LMPDeviceType::array_layout, LMPDeviceType> tdual_efloat_1d;
typedef tdual_efloat_1d::t_dev t_efloat_1d;
typedef tdual_efloat_1d::t_dev_const t_efloat_1d_const;
typedef tdual_efloat_1d::t_dev_um t_efloat_1d_um;
typedef tdual_efloat_1d::t_dev_const_um t_efloat_1d_const_um;
typedef tdual_efloat_1d::t_dev_const_randomread t_efloat_1d_randomread;

//2d E_FLOAT array n*m

typedef Kokkos::DualView<E_FLOAT**, Kokkos::LayoutRight, LMPDeviceType> tdual_efloat_2d;
typedef tdual_efloat_2d::t_dev t_efloat_2d;
typedef tdual_efloat_2d::t_dev_const t_efloat_2d_const;
typedef tdual_efloat_2d::t_dev_um t_efloat_2d_um;
typedef tdual_efloat_2d::t_dev_const_um t_efloat_2d_const_um;
typedef tdual_efloat_2d::t_dev_const_randomread t_efloat_2d_randomread;

//2d E_FLOAT array n*3

typedef Kokkos::DualView<E_FLOAT*[3], Kokkos::LayoutRight, LMPDeviceType> tdual_e_array;
typedef tdual_e_array::t_dev t_e_array;
typedef tdual_e_array::t_dev_const t_e_array_const;
typedef tdual_e_array::t_dev_um t_e_array_um;
typedef tdual_e_array::t_dev_const_um t_e_array_const_um;
typedef tdual_e_array::t_dev_const_randomread t_e_array_randomread;

//Neighbor Types

typedef Kokkos::DualView<int**, LMPDeviceType::array_layout, LMPDeviceType> tdual_neighbors_2d;
typedef tdual_neighbors_2d::t_dev t_neighbors_2d;
typedef tdual_neighbors_2d::t_dev_const t_neighbors_2d_const;
typedef tdual_neighbors_2d::t_dev_um t_neighbors_2d_um;
typedef tdual_neighbors_2d::t_dev_const_um t_neighbors_2d_const_um;
typedef tdual_neighbors_2d::t_dev_const_randomread t_neighbors_2d_randomread;

};

#ifdef LMP_SYCL_GPU
template <>
struct ArrayTypes<LMPHostType> {

//Scalar Types

typedef Kokkos::DualView<int, LMPDeviceType::array_layout, LMPDeviceType> tdual_int_scalar;
typedef tdual_int_scalar::t_host t_int_scalar;
typedef tdual_int_scalar::t_host_const t_int_scalar_const;
typedef tdual_int_scalar::t_host_um t_int_scalar_um;
typedef tdual_int_scalar::t_host_const_um t_int_scalar_const_um;

typedef Kokkos::DualView<LMP_FLOAT, LMPDeviceType::array_layout, LMPDeviceType> tdual_float_scalar;
typedef tdual_float_scalar::t_host t_float_scalar;
typedef tdual_float_scalar::t_host_const t_float_scalar_const;
typedef tdual_float_scalar::t_host_um t_float_scalar_um;
typedef tdual_float_scalar::t_host_const_um t_float_scalar_const_um;

//Generic ArrayTypes
typedef Kokkos::DualView<int*, LMPDeviceType::array_layout, LMPDeviceType> tdual_int_1d;
typedef tdual_int_1d::t_host t_int_1d;
typedef tdual_int_1d::t_host_const t_int_1d_const;
typedef tdual_int_1d::t_host_um t_int_1d_um;
typedef tdual_int_1d::t_host_const_um t_int_1d_const_um;
typedef tdual_int_1d::t_host_const_randomread t_int_1d_randomread;

typedef Kokkos::DualView<int*[3], Kokkos::LayoutRight, LMPDeviceType> tdual_int_1d_3;
typedef tdual_int_1d_3::t_host t_int_1d_3;
typedef tdual_int_1d_3::t_host_const t_int_1d_3_const;
typedef tdual_int_1d_3::t_host_um t_int_1d_3_um;
typedef tdual_int_1d_3::t_host_const_um t_int_1d_3_const_um;
typedef tdual_int_1d_3::t_host_const_randomread t_int_1d_3_randomread;

typedef Kokkos::DualView<int**, Kokkos::LayoutRight, LMPDeviceType> tdual_int_2d;
typedef tdual_int_2d::t_host t_int_2d;
typedef tdual_int_2d::t_host_const t_int_2d_const;
typedef tdual_int_2d::t_host_um t_int_2d_um;
typedef tdual_int_2d::t_host_const_um t_int_2d_const_um;
typedef tdual_int_2d::t_host_const_randomread t_int_2d_randomread;

typedef Kokkos::DualView<int**, LMPDeviceType::array_layout, LMPDeviceType> tdual_int_2d_dl;
typedef tdual_int_2d_dl::t_host t_int_2d_dl;
typedef tdual_int_2d_dl::t_host_const t_int_2d_const_dl;
typedef tdual_int_2d_dl::t_host_um t_int_2d_um_dl;
typedef tdual_int_2d_dl::t_host_const_um t_int_2d_const_um_dl;
typedef tdual_int_2d_dl::t_host_const_randomread t_int_2d_randomread_dl;

typedef Kokkos::DualView<LAMMPS_NS::tagint*, LMPDeviceType::array_layout, LMPDeviceType> tdual_tagint_1d;
typedef tdual_tagint_1d::t_host t_tagint_1d;
typedef tdual_tagint_1d::t_host_const t_tagint_1d_const;
typedef tdual_tagint_1d::t_host_um t_tagint_1d_um;
typedef tdual_tagint_1d::t_host_const_um t_tagint_1d_const_um;
typedef tdual_tagint_1d::t_host_const_randomread t_tagint_1d_randomread;

typedef Kokkos::
  DualView<LAMMPS_NS::tagint**, Kokkos::LayoutRight, LMPDeviceType>
  tdual_tagint_2d;
typedef tdual_tagint_2d::t_host t_tagint_2d;
typedef tdual_tagint_2d::t_host_const t_tagint_2d_const;
typedef tdual_tagint_2d::t_host_um t_tagint_2d_um;
typedef tdual_tagint_2d::t_host_const_um t_tagint_2d_const_um;
typedef tdual_tagint_2d::t_host_const_randomread t_tagint_2d_randomread;

typedef Kokkos::
  DualView<LAMMPS_NS::imageint*, LMPDeviceType::array_layout, LMPDeviceType>
  tdual_imageint_1d;
typedef tdual_imageint_1d::t_host t_imageint_1d;
typedef tdual_imageint_1d::t_host_const t_imageint_1d_const;
typedef tdual_imageint_1d::t_host_um t_imageint_1d_um;
typedef tdual_imageint_1d::t_host_const_um t_imageint_1d_const_um;
typedef tdual_imageint_1d::t_host_const_randomread t_imageint_1d_randomread;

typedef Kokkos::
  DualView<double*, Kokkos::LayoutRight, LMPDeviceType> tdual_double_1d;
typedef tdual_double_1d::t_host t_double_1d;
typedef tdual_double_1d::t_host_const t_double_1d_const;
typedef tdual_double_1d::t_host_um t_double_1d_um;
typedef tdual_double_1d::t_host_const_um t_double_1d_const_um;
typedef tdual_double_1d::t_host_const_randomread t_double_1d_randomread;

typedef Kokkos::
  DualView<double**, Kokkos::LayoutRight, LMPDeviceType> tdual_double_2d;
typedef tdual_double_2d::t_host t_double_2d;
typedef tdual_double_2d::t_host_const t_double_2d_const;
typedef tdual_double_2d::t_host_um t_double_2d_um;
typedef tdual_double_2d::t_host_const_um t_double_2d_const_um;
typedef tdual_double_2d::t_host_const_randomread t_double_2d_randomread;

//1d float array n
typedef Kokkos::DualView<LMP_FLOAT*, LMPDeviceType::array_layout, LMPDeviceType> tdual_float_1d;
typedef tdual_float_1d::t_host t_float_1d;
typedef tdual_float_1d::t_host_const t_float_1d_const;
typedef tdual_float_1d::t_host_um t_float_1d_um;
typedef tdual_float_1d::t_host_const_um t_float_1d_const_um;
typedef tdual_float_1d::t_host_const_randomread t_float_1d_randomread;

//2d float array n
typedef Kokkos::DualView<LMP_FLOAT**, Kokkos::LayoutRight, LMPDeviceType> tdual_float_2d;
typedef tdual_float_2d::t_host t_float_2d;
typedef tdual_float_2d::t_host_const t_float_2d_const;
typedef tdual_float_2d::t_host_um t_float_2d_um;
typedef tdual_float_2d::t_host_const_um t_float_2d_const_um;
typedef tdual_float_2d::t_host_const_randomread t_float_2d_randomread;

//Position Types
//1d X_FLOAT array n
typedef Kokkos::DualView<X_FLOAT*, LMPDeviceType::array_layout, LMPDeviceType> tdual_xfloat_1d;
typedef tdual_xfloat_1d::t_host t_xfloat_1d;
typedef tdual_xfloat_1d::t_host_const t_xfloat_1d_const;
typedef tdual_xfloat_1d::t_host_um t_xfloat_1d_um;
typedef tdual_xfloat_1d::t_host_const_um t_xfloat_1d_const_um;
typedef tdual_xfloat_1d::t_host_const_randomread t_xfloat_1d_randomread;

//2d X_FLOAT array n*m
typedef stdex::mdspan<X_FLOAT, stdex::dextents<std::size_t, 2u>, stdex::layout_right> tdual_xfloat_2d;
typedef Kokkos::DualView<X_FLOAT**, Kokkos::LayoutRight, LMPDeviceType> tdual_xfloat_2d;
typedef tdual_xfloat_2d::t_host t_xfloat_2d;
typedef tdual_xfloat_2d::t_host_const t_xfloat_2d_const;
typedef tdual_xfloat_2d::t_host_um t_xfloat_2d_um;
typedef tdual_xfloat_2d::t_host_const_um t_xfloat_2d_const_um;
typedef tdual_xfloat_2d::t_host_const_randomread t_xfloat_2d_randomread;

//2d X_FLOAT array n*3
typedef Kokkos::DualView<X_FLOAT*[3], Kokkos::LayoutRight, LMPDeviceType> tdual_x_array;
typedef tdual_x_array::t_host t_x_array;
typedef tdual_x_array::t_host_const t_x_array_const;
typedef tdual_x_array::t_host_um t_x_array_um;
typedef tdual_x_array::t_host_const_um t_x_array_const_um;
typedef tdual_x_array::t_host_const_randomread t_x_array_randomread;

//Velocity Types
//1d V_FLOAT array n
typedef Kokkos::DualView<V_FLOAT*, LMPDeviceType::array_layout, LMPDeviceType> tdual_vfloat_1d;
typedef tdual_vfloat_1d::t_host t_vfloat_1d;
typedef tdual_vfloat_1d::t_host_const t_vfloat_1d_const;
typedef tdual_vfloat_1d::t_host_um t_vfloat_1d_um;
typedef tdual_vfloat_1d::t_host_const_um t_vfloat_1d_const_um;
typedef tdual_vfloat_1d::t_host_const_randomread t_vfloat_1d_randomread;

//2d V_FLOAT array n*m
typedef Kokkos::DualView<V_FLOAT**, Kokkos::LayoutRight, LMPDeviceType> tdual_vfloat_2d;
typedef tdual_vfloat_2d::t_host t_vfloat_2d;
typedef tdual_vfloat_2d::t_host_const t_vfloat_2d_const;
typedef tdual_vfloat_2d::t_host_um t_vfloat_2d_um;
typedef tdual_vfloat_2d::t_host_const_um t_vfloat_2d_const_um;
typedef tdual_vfloat_2d::t_host_const_randomread t_vfloat_2d_randomread;

//2d V_FLOAT array n*3
typedef Kokkos::DualView<V_FLOAT*[3], Kokkos::LayoutRight, LMPDeviceType> tdual_v_array;
//typedef Kokkos::DualView<V_FLOAT*[3], LMPDeviceType::array_layout, LMPDeviceType> tdual_v_array;
typedef tdual_v_array::t_host t_v_array;
typedef tdual_v_array::t_host_const t_v_array_const;
typedef tdual_v_array::t_host_um t_v_array_um;
typedef tdual_v_array::t_host_const_um t_v_array_const_um;
typedef tdual_v_array::t_host_const_randomread t_v_array_randomread;

//Force Types
//1d F_FLOAT array n
typedef Kokkos::DualView<F_FLOAT*, LMPDeviceType::array_layout, LMPDeviceType> tdual_ffloat_1d;
typedef tdual_ffloat_1d::t_host t_ffloat_1d;
typedef tdual_ffloat_1d::t_host_const t_ffloat_1d_const;
typedef tdual_ffloat_1d::t_host_um t_ffloat_1d_um;
typedef tdual_ffloat_1d::t_host_const_um t_ffloat_1d_const_um;
typedef tdual_ffloat_1d::t_host_const_randomread t_ffloat_1d_randomread;

//2d F_FLOAT array n*m
typedef Kokkos::DualView<F_FLOAT**, Kokkos::LayoutRight, LMPDeviceType> tdual_ffloat_2d;
typedef tdual_ffloat_2d::t_host t_ffloat_2d;
typedef tdual_ffloat_2d::t_host_const t_ffloat_2d_const;
typedef tdual_ffloat_2d::t_host_um t_ffloat_2d_um;
typedef tdual_ffloat_2d::t_host_const_um t_ffloat_2d_const_um;
typedef tdual_ffloat_2d::t_host_const_randomread t_ffloat_2d_randomread;

//2d F_FLOAT array n*m, device layout
typedef Kokkos::DualView<F_FLOAT**, LMPDeviceType::array_layout, LMPDeviceType> tdual_ffloat_2d_dl;
typedef tdual_ffloat_2d_dl::t_host t_ffloat_2d_dl;
typedef tdual_ffloat_2d_dl::t_host_const t_ffloat_2d_const_dl;
typedef tdual_ffloat_2d_dl::t_host_um t_ffloat_2d_um_dl;
typedef tdual_ffloat_2d_dl::t_host_const_um t_ffloat_2d_const_um_dl;
typedef tdual_ffloat_2d_dl::t_host_const_randomread t_ffloat_2d_randomread_dl;

//2d F_FLOAT array n*3
typedef Kokkos::DualView<F_FLOAT*[3], Kokkos::LayoutRight, LMPDeviceType> tdual_f_array;
//typedef Kokkos::DualView<F_FLOAT*[3], LMPDeviceType::array_layout, LMPDeviceType> tdual_f_array;
typedef tdual_f_array::t_host t_f_array;
typedef tdual_f_array::t_host_const t_f_array_const;
typedef tdual_f_array::t_host_um t_f_array_um;
typedef tdual_f_array::t_host_const_um t_f_array_const_um;
typedef tdual_f_array::t_host_const_randomread t_f_array_randomread;

//2d F_FLOAT array n*6 (for virial)
typedef Kokkos::DualView<F_FLOAT*[6], Kokkos::LayoutRight, LMPDeviceType> tdual_virial_array;
typedef tdual_virial_array::t_host t_virial_array;
typedef tdual_virial_array::t_host_const t_virial_array_const;
typedef tdual_virial_array::t_host_um t_virial_array_um;
typedef tdual_virial_array::t_host_const_um t_virial_array_const_um;
typedef tdual_virial_array::t_host_const_randomread t_virial_array_randomread;


//Energy Types
//1d E_FLOAT array n
typedef Kokkos::DualView<E_FLOAT*, LMPDeviceType::array_layout, LMPDeviceType> tdual_efloat_1d;
typedef tdual_efloat_1d::t_host t_efloat_1d;
typedef tdual_efloat_1d::t_host_const t_efloat_1d_const;
typedef tdual_efloat_1d::t_host_um t_efloat_1d_um;
typedef tdual_efloat_1d::t_host_const_um t_efloat_1d_const_um;
typedef tdual_efloat_1d::t_host_const_randomread t_efloat_1d_randomread;

//2d E_FLOAT array n*m
typedef Kokkos::DualView<E_FLOAT**, Kokkos::LayoutRight, LMPDeviceType> tdual_efloat_2d;
typedef tdual_efloat_2d::t_host t_efloat_2d;
typedef tdual_efloat_2d::t_host_const t_efloat_2d_const;
typedef tdual_efloat_2d::t_host_um t_efloat_2d_um;
typedef tdual_efloat_2d::t_host_const_um t_efloat_2d_const_um;
typedef tdual_efloat_2d::t_host_const_randomread t_efloat_2d_randomread;

//2d E_FLOAT array n*3
typedef Kokkos::DualView<E_FLOAT*[3], Kokkos::LayoutRight, LMPDeviceType> tdual_e_array;
typedef tdual_e_array::t_host t_e_array;
typedef tdual_e_array::t_host_const t_e_array_const;
typedef tdual_e_array::t_host_um t_e_array_um;
typedef tdual_e_array::t_host_const_um t_e_array_const_um;
typedef tdual_e_array::t_host_const_randomread t_e_array_randomread;

//Neighbor Types
typedef Kokkos::DualView<int**, LMPDeviceType::array_layout, LMPDeviceType> tdual_neighbors_2d;
typedef tdual_neighbors_2d::t_host t_neighbors_2d;
typedef tdual_neighbors_2d::t_host_const t_neighbors_2d_const;
typedef tdual_neighbors_2d::t_host_um t_neighbors_2d_um;
typedef tdual_neighbors_2d::t_host_const_um t_neighbors_2d_const_um;
typedef tdual_neighbors_2d::t_host_const_randomread t_neighbors_2d_randomread;

};
#endif
//default LAMMPS Types
typedef struct ArrayTypes<LMPDeviceType> DAT;
typedef struct ArrayTypes<LMPHostType> HAT;

template<class DeviceType, class BufferView, class DualView>
void buffer_view(BufferView &buf, DualView &view,
                 const size_t n0,
                 const size_t n1) {

  buf = BufferView(view.template view<DeviceType>().data(),n0,n1);

}

template<class DeviceType>
struct MemsetZeroFunctor {
  typedef DeviceType  execution_space ;
  void* ptr;
  __attribute__((always_inline)) void operator()(const int i) const {
    ((int*)ptr)[i] = 0;
  }
};

template<class ViewType>
void memset_kokkos (ViewType &view) {
  static MemsetZeroFunctor<typename ViewType::execution_space> f;
  f.ptr = view.data();
  #ifndef SYCL_USING_DEPRECATED_VIEW
  Kokkos::parallel_for(view.span()*sizeof(typename ViewType::value_type)/4, f);
  #else
  Kokkos::parallel_for(view.span()*sizeof(typename ViewType::value_type)/4, f);
  #endif
  ViewType::execution_space().fence();
}

struct params_lj_coul {
  __attribute__((always_inline))
  params_lj_coul() {cut_ljsq=0;cut_coulsq=0;lj1=0;lj2=0;lj3=0;lj4=0;offset=0;};
  __attribute__((always_inline))
  params_lj_coul(int /*i*/) {cut_ljsq=0;cut_coulsq=0;lj1=0;lj2=0;lj3=0;lj4=0;offset=0;};
  F_FLOAT cut_ljsq,cut_coulsq,lj1,lj2,lj3,lj4,offset;
};

// Pair SNAP

#define SNAP_SYCL_REAL double
#define SNAP_SYCL_HOST_VECLEN 1

#ifdef LMP_SYCL_GPU
#define SNAP_SYCL_DEVICE_VECLEN 16
#else
#define SNAP_SYCL_DEVICE_VECLEN 1
#endif


// intentional: SNAreal/complex gets reused beyond SNAP
typedef double SNAreal;

#define LAMMPS_LAMBDA [=]

#ifdef LMP_SYCL_GPU
#if defined(__SYCL_DEVICE_ONLY__)
#define LMP_KK_DEVICE_COMPILE
#endif
#endif

#endif
