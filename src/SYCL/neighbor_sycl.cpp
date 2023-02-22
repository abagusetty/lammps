// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "neighbor_sycl.h"

#include "angle.h"
#include "atom_sycl.h"
#include "atom_masks.h"
#include "bond.h"
#include "comm.h"
#include "dihedral.h"
#include "error.h"
#include "fix.h"
#include "force.h"
#include "improper.h"
#include "kokkos.h"
#include "memory_sycl.h"
#include "neigh_request.h"
#include "pair.h"
#include "style_nbin.h"
#include "style_npair.h"
#include "style_nstencil.h"
#include "style_ntopo.h"
#include "update.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

NeighborSycl::NeighborSycl(LAMMPS *lmp) : Neighbor(lmp),
  neighbond_host(lmp),neighbond_device(lmp)
{
  device_flag = 0;
  bondlist = nullptr;
  anglelist = nullptr;
  dihedrallist = nullptr;
  improperlist = nullptr;
}

/* ---------------------------------------------------------------------- */

NeighborSycl::~NeighborSycl()
{
  if (!copymode) {
    memorySYCL->destroy_sycl(k_cutneighsq,cutneighsq);
    cutneighsq = nullptr;

    memorySYCL->destroy_sycl(k_ex_type,ex_type);
    memorySYCL->destroy_sycl(k_ex1_type,ex1_type);
    memorySYCL->destroy_sycl(k_ex2_type,ex2_type);
    memorySYCL->destroy_sycl(k_ex1_group,ex1_group);
    memorySYCL->destroy_sycl(k_ex2_group,ex2_group);
    memorySYCL->destroy_sycl(k_ex_mol_group,ex_mol_group);
    memorySYCL->destroy_sycl(k_ex1_bit,ex1_bit);
    memorySYCL->destroy_sycl(k_ex2_bit,ex2_bit);
    memorySYCL->destroy_sycl(k_ex_mol_bit,ex_mol_bit);
    memorySYCL->destroy_sycl(k_ex_mol_intra,ex_mol_intra);

    memorySYCL->destroy_sycl(k_bondlist,bondlist);
    memorySYCL->destroy_sycl(k_anglelist,anglelist);
    memorySYCL->destroy_sycl(k_dihedrallist,dihedrallist);
    memorySYCL->destroy_sycl(k_improperlist,improperlist);
  }
}

/* ---------------------------------------------------------------------- */

void NeighborSycl::init()
{
  atomSYCL = (AtomKokkos *) atom;
  Neighbor::init();

  // 1st time allocation of xhold

  if (dist_check)
      xhold = DAT::tdual_x_array("neigh:xhold",maxhold);
}

/* ---------------------------------------------------------------------- */

void NeighborSycl::init_cutneighsq_sycl(int n)
{
  memorySYCL->create_sycl(k_cutneighsq,cutneighsq,n+1,n+1,"neigh:cutneighsq");
  k_cutneighsq.modify<LMPHostType>();
}

/* ---------------------------------------------------------------------- */

void NeighborSycl::create_sycl_list(int i)
{
  if (style != Neighbor::BIN)
    error->all(FLERR,"KOKKOS package only supports 'bin' neighbor lists");

  if (requests[i]->kokkos_device) {
    lists[i] = new NeighListKokkos<LMPDeviceType>(lmp);
    device_flag = 1;
  } else if (requests[i]->kokkos_host)
    lists[i] = new NeighListKokkos<LMPHostType>(lmp);
}

/* ---------------------------------------------------------------------- */

void NeighborSycl::init_ex_type_sycl(int n)
{
  memorySYCL->create_sycl(k_ex_type,ex_type,n+1,n+1,"neigh:ex_type");
  k_ex_type.modify<LMPHostType>();
}

/* ---------------------------------------------------------------------- */

void NeighborSycl::init_ex_bit_sycl()
{
  memorySYCL->create_sycl(k_ex1_bit, ex1_bit, nex_group, "neigh:ex1_bit");
  k_ex1_bit.modify<LMPHostType>();
  memorySYCL->create_sycl(k_ex2_bit, ex2_bit, nex_group, "neigh:ex2_bit");
  k_ex2_bit.modify<LMPHostType>();
}

/* ---------------------------------------------------------------------- */

void NeighborSycl::init_ex_mol_bit_sycl()
{
  memorySYCL->create_sycl(k_ex_mol_bit, ex_mol_bit, nex_mol, "neigh:ex_mol_bit");
  k_ex_mol_bit.modify<LMPHostType>();
}

/* ---------------------------------------------------------------------- */

void NeighborSycl::grow_ex_mol_intra_sycl()
{
  memorySYCL->grow_sycl(k_ex_mol_intra, ex_mol_intra, maxex_mol, "neigh:ex_mol_intra");
  k_ex_mol_intra.modify<LMPHostType>();
}

/* ----------------------------------------------------------------------
   if any atom moved trigger distance (half of neighbor skin) return 1
   shrink trigger distance if box size has changed
   conservative shrink procedure:
     compute distance each of 8 corners of box has moved since last reneighbor
     reduce skin distance by sum of 2 largest of the 8 values
     new trigger = 1/2 of reduced skin distance
   for orthogonal box, only need 2 lo/hi corners
   for triclinic, need all 8 corners since deformations can displace all 8
------------------------------------------------------------------------- */

int NeighborSycl::check_distance()
{
  if (device_flag)
    return check_distance_sycl<LMPDeviceType>();
  else
    return check_distance_sycl<LMPHostType>();
}

template<class DeviceType>
int NeighborSycl::check_distance_sycl()
{
  double delx,dely,delz;
  double delta,delta1,delta2;

  if (boxcheck) {
    if (triclinic == 0) {
      delx = bboxlo[0] - boxlo_hold[0];
      dely = bboxlo[1] - boxlo_hold[1];
      delz = bboxlo[2] - boxlo_hold[2];
      delta1 = sqrt(delx*delx + dely*dely + delz*delz);
      delx = bboxhi[0] - boxhi_hold[0];
      dely = bboxhi[1] - boxhi_hold[1];
      delz = bboxhi[2] - boxhi_hold[2];
      delta2 = sqrt(delx*delx + dely*dely + delz*delz);
      delta = 0.5 * (skin - (delta1+delta2));
      deltasq = delta*delta;
    } else {
      domain->box_corners();
      delta1 = delta2 = 0.0;
      for (int i = 0; i < 8; i++) {
        delx = corners[i][0] - corners_hold[i][0];
        dely = corners[i][1] - corners_hold[i][1];
        delz = corners[i][2] - corners_hold[i][2];
        delta = sqrt(delx*delx + dely*dely + delz*delz);
        if (delta > delta1) delta1 = delta;
        else if (delta > delta2) delta2 = delta;
      }
      delta = 0.5 * (skin - (delta1+delta2));
      deltasq = delta*delta;
    }
  } else deltasq = triggersq;

  atomSYCL->sync(ExecutionSpaceFromDevice<DeviceType>::space,X_MASK);
  x = atomSYCL->k_x;
  xhold.sync<DeviceType>();
  int nlocal = atom->nlocal;
  if (includegroup) nlocal = atom->nfirst;

  int flag = 0;
  copymode = 1;
  Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagNeighborCheckDistance<DeviceType> >(0,nlocal),*this,flag);
  copymode = 0;

  int flagall;
  MPI_Allreduce(&flag,&flagall,1,MPI_INT,MPI_MAX,world);
  if (flagall && ago == MAX(every,delay)) ndanger++;
  return flagall;
}

template<class DeviceType>
__attribute__((always_inline))
void NeighborSycl::operator()(TagNeighborCheckDistance<DeviceType>, const int &i, int &flag) const {
  const X_FLOAT delx = x.view<DeviceType>()(i,0) - xhold.view<DeviceType>()(i,0);
  const X_FLOAT dely = x.view<DeviceType>()(i,1) - xhold.view<DeviceType>()(i,1);
  const X_FLOAT delz = x.view<DeviceType>()(i,2) - xhold.view<DeviceType>()(i,2);
  const X_FLOAT rsq = delx*delx + dely*dely + delz*delz;
  if (rsq > deltasq) flag = 1;
}

/* ----------------------------------------------------------------------
   build perpetuals neighbor lists
   called at setup and every few timesteps during run or minimization
   topology lists also built if topoflag = 1, CUDA calls with topoflag = 0
------------------------------------------------------------------------- */


void NeighborSycl::build(int topoflag)
{
  if (device_flag)
    build_sycl<LMPDeviceType>(topoflag);
  else
    build_sycl<LMPHostType>(topoflag);
}

template<class DeviceType>
void NeighborSycl::build_sycl(int topoflag)
{
  int i,m;

  ago = 0;
  ncalls++;
  lastcall = update->ntimestep;

  int nlocal = atom->nlocal;
  int nall = nlocal + atom->nghost;

  // check that using special bond flags will not overflow neigh lists

  if (nall > NEIGHMASK)
    error->one(FLERR,"Too many local+ghost atoms for neighbor list");

  // store current atom positions and box size if needed

  if (dist_check) {
    atomSYCL->sync(ExecutionSpaceFromDevice<DeviceType>::space,X_MASK);
    x = atomSYCL->k_x;
    if (includegroup) nlocal = atom->nfirst;
    int maxhold_sycl = xhold.view<DeviceType>().extent(0);
    if (atom->nmax > maxhold || maxhold_sycl < maxhold) {
      maxhold = atom->nmax;
      xhold = DAT::tdual_x_array("neigh:xhold",maxhold);
    }
    copymode = 1;
    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagNeighborXhold<DeviceType> >(0,nlocal),*this);
    copymode = 0;
    xhold.modify<DeviceType>();
    if (boxcheck) {
      if (triclinic == 0) {
        boxlo_hold[0] = bboxlo[0];
        boxlo_hold[1] = bboxlo[1];
        boxlo_hold[2] = bboxlo[2];
        boxhi_hold[0] = bboxhi[0];
        boxhi_hold[1] = bboxhi[1];
        boxhi_hold[2] = bboxhi[2];
      } else {
        domain->box_corners();
        corners = domain->corners;
        for (i = 0; i < 8; i++) {
          corners_hold[i][0] = corners[i][0];
          corners_hold[i][1] = corners[i][1];
          corners_hold[i][2] = corners[i][2];
        }
      }
    }
  }

  // bin atoms for all NBin instances
  // not just NBin associated with perpetual lists
  // b/c cannot wait to bin occasional lists in build_one() call
  // if bin then, atoms may have moved outside of proc domain & bin extent,
  //   leading to errors or even a crash

  if (style != Neighbor::NSQ) {
    for (int i = 0; i < nbin; i++) {
      if (!neigh_bin[i]->kokkos) atomSYCL->sync(Host,ALL_MASK);
      neigh_bin[i]->bin_atoms_setup(nall);
      neigh_bin[i]->bin_atoms();
    }
  }

  // build pairwise lists for all perpetual NPair/NeighList
  // grow() with nlocal/nall args so that only realloc if have to

  for (i = 0; i < npair_perpetual; i++) {
    m = plist[i];
    if (!lists[m]->kokkos) atomSYCL->sync(Host,ALL_MASK);
    if (!lists[m]->copy) lists[m]->grow(nlocal,nall);
    neigh_pair[m]->build_setup();
    neigh_pair[m]->build(lists[m]);
  }

  // build topology lists for bonds/angles/etc

  if ((atom->molecular != Atom::ATOMIC) && topoflag) build_topology();
}

template<class DeviceType>
__attribute__((always_inline))
void NeighborSycl::operator()(TagNeighborXhold<DeviceType>, const int &i) const {
  xhold.view<DeviceType>()(i,0) = x.view<DeviceType>()(i,0);
  xhold.view<DeviceType>()(i,1) = x.view<DeviceType>()(i,1);
  xhold.view<DeviceType>()(i,2) = x.view<DeviceType>()(i,2);
}

/* ---------------------------------------------------------------------- */

void NeighborSycl::modify_ex_type_grow_sycl() {
  memorySYCL->grow_sycl(k_ex1_type,ex1_type,maxex_type,"neigh:ex1_type");
  k_ex1_type.modify<LMPHostType>();
  memorySYCL->grow_sycl(k_ex2_type,ex2_type,maxex_type,"neigh:ex2_type");
  k_ex2_type.modify<LMPHostType>();
}

/* ---------------------------------------------------------------------- */
void NeighborSycl::modify_ex_group_grow_sycl() {
  memorySYCL->grow_sycl(k_ex1_group,ex1_group,maxex_group,"neigh:ex1_group");
  k_ex1_group.modify<LMPHostType>();
  memorySYCL->grow_sycl(k_ex2_group,ex2_group,maxex_group,"neigh:ex2_group");
  k_ex2_group.modify<LMPHostType>();
}

/* ---------------------------------------------------------------------- */
void NeighborSycl::modify_mol_group_grow_sycl() {
  memorySYCL->grow_sycl(k_ex_mol_group,ex_mol_group,maxex_mol,"neigh:ex_mol_group");
  k_ex_mol_group.modify<LMPHostType>();
}

/* ---------------------------------------------------------------------- */
void NeighborSycl::modify_mol_intra_grow_sycl() {
  memorySYCL->grow_sycl(k_ex_mol_intra,ex_mol_intra,maxex_mol,"neigh:ex_mol_intra");
  k_ex_mol_intra.modify<LMPHostType>();
}

/* ---------------------------------------------------------------------- */
void NeighborSycl::set_binsize_sycl() {
  if (!binsizeflag && lmp->kokkos->ngpus > 0) {
    binsize_user = cutneighmax;
    binsizeflag = 1;
  }
}

/* ---------------------------------------------------------------------- */

void NeighborSycl::init_topology() {
  if (device_flag) {
    neighbond_device.init_topology_sycl();
  } else {
    neighbond_host.init_topology_sycl();
  }
}

/* ----------------------------------------------------------------------
   build all topology neighbor lists every few timesteps
   normally built with pair lists, but CUDA separates them
------------------------------------------------------------------------- */

void NeighborSycl::build_topology() {
  if (device_flag) {
    neighbond_device.build_topology_sycl();

    k_bondlist = neighbond_device.k_bondlist;
    k_anglelist = neighbond_device.k_anglelist;
    k_dihedrallist = neighbond_device.k_dihedrallist;
    k_improperlist = neighbond_device.k_improperlist;

    // Transfer topology neighbor lists to Host for non-Kokkos styles

    if (force->bond && force->bond->execution_space == Host)
      k_bondlist.sync<LMPHostType>();
    if (force->angle && force->angle->execution_space == Host)
      k_anglelist.sync<LMPHostType>();
    if (force->dihedral && force->dihedral->execution_space == Host)
      k_dihedrallist.sync<LMPHostType>();
    if (force->improper && force->improper->execution_space == Host)
      k_improperlist.sync<LMPHostType>();

   } else {
    neighbond_host.build_topology_sycl();

    k_bondlist = neighbond_host.k_bondlist;
    k_anglelist = neighbond_host.k_anglelist;
    k_dihedrallist = neighbond_host.k_dihedrallist;
    k_improperlist = neighbond_host.k_improperlist;
  }
}
