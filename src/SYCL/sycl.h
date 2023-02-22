// clang-format off
/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifndef SYCL_LMP_H
#define SYCL_LMP_H

#include "pointers.h"
#include "sycl_type.h"
#include "pair_sycl.h"

namespace LAMMPS_NS {

class SyclLMP : protected Pointers {
 public:
  int sycl_exists;
  int neighflag;
  int neighflag_qeq;
  int neighflag_qeq_set;
  int exchange_comm_classic;
  int forward_comm_classic;
  int forward_pair_comm_classic;
  int reverse_pair_comm_classic;
  int forward_fix_comm_classic;
  int reverse_comm_classic;
  int exchange_comm_on_host;
  int forward_comm_on_host;
  int reverse_comm_on_host;
  int exchange_comm_changed;
  int forward_comm_changed;
  int forward_pair_comm_changed;
  int reverse_pair_comm_changed;
  int forward_fix_comm_changed;
  int reverse_comm_changed;
  int nthreads,ngpus;
  int auto_sync;
  int gpu_aware_flag;
  int neigh_thread;
  int neigh_thread_set;
  int neigh_transpose;
  int newtonflag;
  int allow_overlap;
  double binsize;

  static int is_finalized;
  static int init_ngpus;

  SyclLMP(class LAMMPS *, int, char **);

  static void initialize(const Sycl::InitializationSettings&, Error *);
  static void finalize();
  void accelerator(int, char **);
  int neigh_count(int);

  // TODO: ABB: check this function and also the ScatterDuplicated stuff
  template<class DeviceType>
  int need_dup(int qeq_flag = 0)
  {
    int value = 0;
    int neighflag = this->neighflag;
    if (qeq_flag) neighflag = this->neighflag_qeq;

    if (neighflag == HALFTHREAD)
      value = std::is_same<typename NeedDup<HALFTHREAD,DeviceType>::value,Sycl::Experimental::ScatterDuplicated>::value;

    return value;
  }

 private:
  static void my_signal_handler(int);
};

}

#endif

