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

#ifndef LMP_SYCL_BASE_FFT_H
#define LMP_SYCL_BASE_FFT_H

#include "fftdata_sycl.h"

namespace LAMMPS_NS {

class SyclBaseFFT {
 public:
  SyclBaseFFT() {}

  //Kspace
  virtual void pack_forward_grid_sycl(int, FFT_DAT::tdual_FFT_SCALAR_1d &, int, DAT::tdual_int_2d &, int) {};
  virtual void unpack_forward_grid_sycl(int, FFT_DAT::tdual_FFT_SCALAR_1d &, int, int, DAT::tdual_int_2d &, int) {};
  virtual void pack_reverse_grid_sycl(int, FFT_DAT::tdual_FFT_SCALAR_1d &, int, DAT::tdual_int_2d &, int) {};
  virtual void unpack_reverse_grid_sycl(int, FFT_DAT::tdual_FFT_SCALAR_1d &, int, int, DAT::tdual_int_2d &, int) {};
};

}

#endif

