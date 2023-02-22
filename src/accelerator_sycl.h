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

#ifndef LMP_ACCELERATOR_SYCL_H
#define LMP_ACCELERATOR_SYCL_H

// true interface to SYCL
// used when SYCL is installed

#ifdef LMP_SYCL

#include "atom_sycl.h"          // IWYU pragma: export
#include "comm_sycl.h"          // IWYU pragma: export
#include "comm_tiled_sycl.h"    // IWYU pragma: export
#include "domain_sycl.h"        // IWYU pragma: export
#include "sycl.h"               // IWYU pragma: export
#include "memory_sycl.h"        // IWYU pragma: export
#include "modify_sycl.h"        // IWYU pragma: export
#include "neighbor_sycl.h"      // IWYU pragma: export

#define LAMMPS_INLINE __attribute__((always_inline))

#else

// dummy interface to SYCL
// needed for compiling when SYCL is not installed

#include "atom.h"
#include "comm_brick.h"
#include "comm_tiled.h"
#include "domain.h"
#include "memory.h"
#include "modify.h"
#include "neighbor.h"

#define LAMMPS_INLINE inline

namespace LAMMPS_NS {

class SyclLMP {
 public:
  int sycl_exists;
  int nthreads;
  int ngpus;

  SyclLMP(class LAMMPS *, int, char **) { sycl_exists = 0; }
  ~SyclLMP() {}
  static void finalize() {}
  void accelerator(int, char **) {}
  int neigh_list_sycl(int) { return 0; }
  int neigh_count(int) { return 0; }
};

class AtomSycl : public Atom {
 public:
  tagint **k_special;
  AtomSycl(class LAMMPS *lmp) : Atom(lmp) {}
  void sync(const ExecutionSpace /*space*/, unsigned int /*mask*/) {}
  void modified(const ExecutionSpace /*space*/, unsigned int /*mask*/) {}
};

class CommSycl : public CommBrick {
 public:
  CommSycl(class LAMMPS *lmp) : CommBrick(lmp) {}
};

class CommTiledSycl : public CommTiled {
 public:
  CommTiledSycl(class LAMMPS *lmp) : CommTiled(lmp) {}
  CommTiledSycl(class LAMMPS *lmp, Comm *oldcomm) : CommTiled(lmp, oldcomm) {}
};

class DomainSycl : public Domain {
 public:
  DomainSycl(class LAMMPS *lmp) : Domain(lmp) {}
};

class NeighborSycl : public Neighbor {
 public:
  NeighborSycl(class LAMMPS *lmp) : Neighbor(lmp) {}
};

class MemorySycl : public Memory {
 public:
  MemorySycl(class LAMMPS *lmp) : Memory(lmp) {}
  void grow_sycl(tagint **, tagint **, int, int, const char *) {}
};

class ModifySycl : public Modify {
 public:
  ModifySycl(class LAMMPS *lmp) : Modify(lmp) {}
};

class DAT {
 public:
  typedef double tdual_xfloat_1d;
  typedef double tdual_FFT_SCALAR_1d;
  typedef int tdual_int_1d;
  typedef int tdual_int_2d;
};

}    // namespace LAMMPS_NS

#endif
#endif
