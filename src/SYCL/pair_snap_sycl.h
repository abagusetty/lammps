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

#ifdef PAIR_CLASS
// clang-format off
PairStyle(snap/kk,PairSNAPSYCLDevice<LMPDeviceType>);
PairStyle(snap/kk/device,PairSNAPSYCLDevice<LMPDeviceType>);
#ifdef LMP_SYCL_GPU
PairStyle(snap/kk/host,PairSNAPSYCLHost<LMPHostType>);
#else
PairStyle(snap/kk/host,PairSNAPSYCLDevice<LMPHostType>);
#endif
// clang-format on
#else

// clang-format off
#ifndef LMP_PAIR_SNAP_SYCL_H
#define LMP_PAIR_SNAP_SYCL_H

#include "pair_snap.h"
#include "sycl_type.h"
#include "neigh_list_kokkos.h"
#include "sna_kokkos.h"
#include "pair_kokkos.h"

namespace LAMMPS_NS {

// Routines for both the CPU and GPU backend
template<int NEIGHFLAG, int EVFLAG>
struct TagPairSNAPComputeForce{};


// GPU backend only
struct TagPairSNAPComputeNeigh{};
struct TagPairSNAPComputeCayleyKlein{};
struct TagPairSNAPPreUi{};
struct TagPairSNAPComputeUiSmall{}; // more parallelism, more divergence
struct TagPairSNAPComputeUiLarge{}; // less parallelism, no divergence
struct TagPairSNAPTransformUi{}; // re-order ulisttot from SoA to AoSoA, zero ylist
struct TagPairSNAPComputeZi{};
struct TagPairSNAPBeta{};
struct TagPairSNAPComputeBi{};
struct TagPairSNAPTransformBi{}; // re-order blist from AoSoA to AoS
struct TagPairSNAPComputeYi{};
struct TagPairSNAPComputeYiWithZlist{};
template<int dir>
struct TagPairSNAPComputeFusedDeidrjSmall{}; // more parallelism, more divergence
template<int dir>
struct TagPairSNAPComputeFusedDeidrjLarge{}; // less parallelism, no divergence

// CPU backend only
struct TagPairSNAPComputeNeighCPU{};
struct TagPairSNAPPreUiCPU{};
struct TagPairSNAPComputeUiCPU{};
struct TagPairSNAPTransformUiCPU{};
struct TagPairSNAPComputeZiCPU{};
struct TagPairSNAPBetaCPU{};
struct TagPairSNAPComputeBiCPU{};
struct TagPairSNAPZeroYiCPU{};
struct TagPairSNAPComputeYiCPU{};
struct TagPairSNAPComputeDuidrjCPU{};
struct TagPairSNAPComputeDeidrjCPU{};

template<class DeviceType, typename real_type_, int vector_length_>
class PairSNAPSYCL : public PairSNAP {
public:
  enum {EnabledNeighFlags=FULL|HALF|HALFTHREAD};
  enum {COUL_FLAG=0};
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  typedef EV_FLOAT value_type;

  static constexpr int vector_length = vector_length_;
  using real_type = real_type_;
  using complex = SNAComplex<real_type>;

  // Static team/tile sizes for device offload
  static constexpr int team_size_compute_neigh = 4;
  static constexpr int tile_size_compute_ck = 4;
  static constexpr int tile_size_pre_ui = 4;
  static constexpr int team_size_compute_ui = sizeof(real_type) == 4 ? 8 : 4;
  static constexpr int tile_size_transform_ui = 4;
  static constexpr int tile_size_compute_zi = 8;
  static constexpr int tile_size_compute_bi = 4;
  static constexpr int tile_size_transform_bi = 4;
  static constexpr int tile_size_compute_yi = 8;
  static constexpr int team_size_compute_fused_deidrj = sizeof(real_type) == 4 ? 4 : 2;

  // Custom MDRangePolicy, Rank3, to reduce verbosity of kernel launches
  // This hides the Kokkos::IndexType<int> and Kokkos::Rank<3...>
  // and reduces the verbosity of the LaunchBound by hiding the explicit
  // multiplication by vector_length
  template <class Device, int num_tiles, class TagPairSNAP>
  using Snap3DRangePolicy = typename Kokkos::MDRangePolicy<Device, Kokkos::IndexType<int>, Kokkos::Rank<3, Kokkos::Iterate::Left, Kokkos::Iterate::Left>, Kokkos::LaunchBounds<vector_length * num_tiles>, TagPairSNAP>;

  // Custom SnapAoSoATeamPolicy to reduce the verbosity of kernel launches
  // This hides the LaunchBounds abstraction by hiding the explicit
  // multiplication by vector length
  template <class Device, int num_teams, class TagPairSNAP>
  using SnapAoSoATeamPolicy = typename Kokkos::TeamPolicy<Device, Kokkos::LaunchBounds<vector_length * num_teams>, TagPairSNAP>;

  PairSNAPSycl(class LAMMPS *);
  ~PairSNAPSycl();

  void coeff(int, char**);
  void init_style();
  double init_one(int, int);
  void compute(int, int);
  double memory_usage();

  template<class TagStyle>
  void check_team_size_for(int, int&);

  template<class TagStyle>
  void check_team_size_reduce(int, int&);

  template<int NEIGHFLAG, int EVFLAG>
  __attribute__((always_inline))
  void operator() (TagPairSNAPComputeForce<NEIGHFLAG,EVFLAG>,const int& ii) const;

  template<int NEIGHFLAG, int EVFLAG>
  __attribute__((always_inline))
  void operator() (TagPairSNAPComputeForce<NEIGHFLAG,EVFLAG>,const int& ii, EV_FLOAT&) const;

  __attribute__((always_inline))
  void operator() (TagPairSNAPBetaCPU,const int& ii) const;

  // GPU backend only
  __attribute__((always_inline))
  void operator() (TagPairSNAPComputeNeigh,const typename Kokkos::TeamPolicy<DeviceType, TagPairSNAPComputeNeigh>::member_type& team) const;

  __attribute__((always_inline))
  void operator() (TagPairSNAPComputeCayleyKlein, const int iatom_mod, const int jnbor, const int iatom_div) const;

  __attribute__((always_inline))
  void operator() (TagPairSNAPPreUi,const int iatom_mod, const int j, const int iatom_div) const;

  __attribute__((always_inline))
  void operator() (TagPairSNAPComputeUiSmall,const typename Kokkos::TeamPolicy<DeviceType, TagPairSNAPComputeUiSmall>::member_type& team) const;

  __attribute__((always_inline))
  void operator() (TagPairSNAPComputeUiLarge,const typename Kokkos::TeamPolicy<DeviceType, TagPairSNAPComputeUiLarge>::member_type& team) const;

  __attribute__((always_inline))
  void operator() (TagPairSNAPTransformUi,const int iatom_mod, const int j, const int iatom_div) const;

  __attribute__((always_inline))
  void operator() (TagPairSNAPComputeZi,const int iatom_mod, const int idxz, const int iatom_div) const;

  __attribute__((always_inline))
  void operator() (TagPairSNAPBeta, const int& ii) const;

  __attribute__((always_inline))
  void operator() (TagPairSNAPComputeBi,const int iatom_mod, const int idxb, const int iatom_div) const;

  __attribute__((always_inline))
  void operator() (TagPairSNAPTransformBi,const int iatom_mod, const int idxb, const int iatom_div) const;

  __attribute__((always_inline))
  void operator() (TagPairSNAPComputeYi,const int iatom_mod, const int idxz, const int iatom_div) const;

  __attribute__((always_inline))
  void operator() (TagPairSNAPComputeYiWithZlist,const int iatom_mod, const int idxz, const int iatom_div) const;

  template<int dir>
  __attribute__((always_inline))
  void operator() (TagPairSNAPComputeFusedDeidrjSmall<dir>,const typename Kokkos::TeamPolicy<DeviceType, TagPairSNAPComputeFusedDeidrjSmall<dir> >::member_type& team) const;

  template<int dir>
  __attribute__((always_inline))
  void operator() (TagPairSNAPComputeFusedDeidrjLarge<dir>,const typename Kokkos::TeamPolicy<DeviceType, TagPairSNAPComputeFusedDeidrjLarge<dir> >::member_type& team) const;

  // CPU backend only
  __attribute__((always_inline))
  void operator() (TagPairSNAPComputeNeighCPU,const typename Kokkos::TeamPolicy<DeviceType, TagPairSNAPComputeNeighCPU>::member_type& team) const;

  __attribute__((always_inline))
  void operator() (TagPairSNAPPreUiCPU,const typename Kokkos::TeamPolicy<DeviceType, TagPairSNAPPreUiCPU>::member_type& team) const;

  __attribute__((always_inline))
  void operator() (TagPairSNAPComputeUiCPU,const typename Kokkos::TeamPolicy<DeviceType, TagPairSNAPComputeUiCPU>::member_type& team) const;

  __attribute__((always_inline))
  void operator() (TagPairSNAPTransformUiCPU, const int j, const int iatom) const;

  __attribute__((always_inline))
  void operator() (TagPairSNAPComputeZiCPU,const int& ii) const;

  __attribute__((always_inline))
  void operator() (TagPairSNAPComputeBiCPU,const typename Kokkos::TeamPolicy<DeviceType, TagPairSNAPComputeBiCPU>::member_type& team) const;

  __attribute__((always_inline))
  void operator() (TagPairSNAPComputeYiCPU,const int& ii) const;

  __attribute__((always_inline))
  void operator() (TagPairSNAPComputeDuidrjCPU,const typename Kokkos::TeamPolicy<DeviceType, TagPairSNAPComputeDuidrjCPU>::member_type& team) const;

  __attribute__((always_inline))
  void operator() (TagPairSNAPComputeDeidrjCPU,const typename Kokkos::TeamPolicy<DeviceType, TagPairSNAPComputeDeidrjCPU>::member_type& team) const;

  template<int NEIGHFLAG>
  __attribute__((always_inline))
  void v_tally_xyz(EV_FLOAT &ev, const int &i, const int &j,
      const F_FLOAT &fx, const F_FLOAT &fy, const F_FLOAT &fz,
      const F_FLOAT &delx, const F_FLOAT &dely, const F_FLOAT &delz) const;

protected:
  typename AT::t_neighbors_2d d_neighbors;
  typename AT::t_int_1d_randomread d_ilist;
  typename AT::t_int_1d_randomread d_numneigh;

  DAT::tdual_efloat_1d k_eatom;
  DAT::tdual_virial_array k_vatom;
  typename AT::t_efloat_1d d_eatom;
  typename AT::t_virial_array d_vatom;

  SNAKokkos<DeviceType, real_type, vector_length> snaKK;

  int inum,max_neighs,chunk_size,chunk_offset;
  int host_flag;

  int eflag,vflag;

  void allocate() override;

  stdex::mdspan<real_type*,   stdex::dextents<std::size_t, 1u>, stdex::layout_right> d_radelem;     // element radii
  stdex::mdspan<real_type*,   stdex::dextents<std::size_t, 1u>, stdex::layout_right> d_wjelem;      // elements weights
  stdex::mdspan<real_type**,  stdex::dextents<std::size_t, 1u>, stdex::layout_right> d_coeffelem;   // element bispectrum coefficients
  stdex::mdspan<real_type*,   stdex::dextents<std::size_t, 1u>, stdex::layout_right> d_sinnerelem;  // element inner cutoff midpoint
  stdex::mdspan<real_type*,   stdex::dextents<std::size_t, 1u>, stdex::layout_right> d_dinnerelem;  // element inner cutoff half-width
  stdex::mdspan<T_INT*,       stdex::dextents<std::size_t, 1u>, stdex::layout_right> d_map;         // mapping from atom types to elements
  stdex::mdspan<T_INT*,       stdex::dextents<std::size_t, 1u>, stdex::layout_right> d_ninside;     // ninside for all atoms in list
  stdex::mdspan<real_type**,  stdex::dextents<std::size_t, 1u>, stdex::layout_right> d_beta;        // betas for all atoms in list
  stdex::mdspan<real_type***, stdex::dextents<std::size_t, 1u>, stdex::layout_left > d_beta_pack;   // betas for all atoms in list, GPU

  typedef Kokkos::DualView<F_FLOAT**, DeviceType> tdual_fparams;
  tdual_fparams k_cutsq;
  typedef Kokkos::View<const F_FLOAT**, DeviceType, Kokkos::MemoryTraits<Kokkos::RandomAccess> > t_fparams_rnd;
  t_fparams_rnd rnd_cutsq;

  typename AT::t_x_array_randomread x;
  typename AT::t_f_array f;
  typename AT::t_int_1d_randomread type;

  int need_dup;

  // ABB: 05/25/22 working here:
  // TODO: need to port Kokkos-based variables like dup_f, dup_vatom, ndup_f, nduP_vatom to SYCL
  F_FLOAT*[3] dup_f;
  F_FLOAT*[6] dup_vatom;
  F_FLOAT*[3] ndup_f;
  F_FLOAT*[6] ndup_vatom;


  using KKDeviceType = typename KKDevice<DeviceType>::value;

  template<typename DataType, typename Layout>
  using DupScatterView = KKScatterView<DataType, Layout, KKDeviceType, KKScatterSum, KKScatterDuplicated>;

  template<typename DataType, typename Layout>
  using NonDupScatterView = KKScatterView<DataType, Layout, KKDeviceType, KKScatterSum, KKScatterNonDuplicated>;

  DupScatterView<F_FLOAT*[3], typename DAT::t_f_array::array_layout> dup_f;
  DupScatterView<F_FLOAT*[6], typename DAT::t_virial_array::array_layout> dup_vatom;

  NonDupScatterView<F_FLOAT*[3], typename DAT::t_f_array::array_layout> ndup_f;
  NonDupScatterView<F_FLOAT*[6], typename DAT::t_virial_array::array_layout> ndup_vatom;

  friend void pair_virial_fdotr_compute<PairSNAPSycl>(PairSNAPSycl*);

  // Utility routine which wraps computing per-team scratch size requirements for
  // ComputeNeigh, ComputeUi, and ComputeFusedDeidrj
  template <typename scratch_type>
  int scratch_size_helper(int values_per_team);

};


// These wrapper classes exist to make the pair style factory happy/avoid having
// to extend the pair style factory to support Pair classes w/an arbitrary number
// of extra template parameters

template <class DeviceType>
class PairSNAPSyclDevice : public PairSNAPSycl<DeviceType, SNAP_SYCL_REAL, SNAP_SYCL_DEVICE_VECLEN> {

private:
  using Base = PairSNAPSycl<DeviceType, SNAP_SYCL_REAL, SNAP_SYCL_DEVICE_VECLEN>;

public:

  PairSNAPSyclDevice(class LAMMPS *);

  void coeff(int, char**);
  void init_style();
  double init_one(int, int);
  void compute(int, int);
  double memory_usage();

};

#ifdef LMP_SYCL_GPU
template <class DeviceType>
class PairSNAPSyclHost : public PairSNAPSycl<DeviceType, SNAP_SYCL_REAL, SNAP_SYCL_HOST_VECLEN> {

private:
  using Base = PairSNAPSycl<DeviceType, SNAP_SYCL_REAL, SNAP_SYCL_HOST_VECLEN>;

public:

  PairSNAPSyclHost(class LAMMPS *);

  void coeff(int, char**);
  void init_style();
  double init_one(int, int);
  void compute(int, int);
  double memory_usage();

};
#endif

}

#endif
#endif
