// clang-format off
/* -*- c++ -*- ----------------------------------------------------------
   lammps - large-scale atomic/molecular massively parallel simulator
   https://www.lammps.org/, sandia national laboratories
   steve plimpton, sjplimp@sandia.gov

   copyright (2003) sandia corporation.  under the terms of contract
   de-ac04-94al85000 with sandia corporation, the u.s. government retains
   certain rights in this software.  this software is distributed under
   the gnu general public license.

   see the readme file in the top-level lammps directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   contributing authors: christian trott (snl), stan moore (snl),
                         evan weinberg (nvidia)
------------------------------------------------------------------------- */

#include <cmath>
#include <cstdlib>
#include <cstring>
#include "pair_snap_kokkos.h"
#include "atom_kokkos.h"
#include "error.h"
#include "force.h"
#include "atom_masks.h"
#include "memory_kokkos.h"
#include "neigh_request.h"
#include "neighbor_kokkos.h"
#include "kokkos.h"
#include "sna.h"
#include "comm.h"

#define maxline 1024
#define maxword 3

namespace lammps_ns {

// outstanding issues with quadratic term
// 1. there seems to a problem with compute_optimized energy calc
// it does not match compute_regular, even when quadratic coeffs = 0

//static double t1 = 0.0;
//static double t2 = 0.0;
//static double t3 = 0.0;
//static double t4 = 0.0;
//static double t5 = 0.0;
//static double t6 = 0.0;
//static double t7 = 0.0;
/* ---------------------------------------------------------------------- */

template<class typename real_type, int vector_length>
pairsnapsycl<real_type, vector_length>::pairsnapsycl(lammps *lmp) : pairsnap(lmp)
{
  respa_enable = 0;

  kokkosable = 1;
  atomkk = (atomkokkos *) atom;
  execution_space = executionspacefromdevice<devicetype>::space;
  datamask_read = empty_mask;
  datamask_modify = empty_mask;

  k_cutsq = tdual_fparams("pairsnapsycl::cutsq",atom->ntypes+1,atom->ntypes+1);
  auto d_cutsq = k_cutsq.template view<devicetype>();
  rnd_cutsq = d_cutsq;

  host_flag = (execution_space == host);
}

/* ---------------------------------------------------------------------- */

template<class typename real_type, int vector_length>
pairsnapsycl<real_type, vector_length>::~pairsnapsycl()
{
  if (copymode) return;

  memorykk->destroy_kokkos(k_eatom,eatom);
  memorykk->destroy_kokkos(k_vatom,vatom);
}


/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

template<class typename real_type, int vector_length>
void pairsnapsycl<real_type, vector_length>::init_style()
{
  if (host_flag) {
    if (lmp->kokkos->nthreads > 1)
      if (comm->me == 0)
        utils::logmesg(lmp,"pair style snap/kk currently only runs on a single "
                           "cpu thread, even if more threads are requested\n");

    pairsnap::init_style();
    return;
  }

  if (force->newton_pair == 0)
    error->all(flerr,"pair style snap requires newton pair on");

  // irequest = neigh request made by parent class

  neighflag = lmp->kokkos->neighflag;
  int irequest = neighbor->request(this,instance_me);

  neighbor->requests[irequest]->
    kokkos_host = std::is_same<devicetype,lmphosttype>::value &&
    !std::is_same<devicetype,lmpdevicetype>::value;
  neighbor->requests[irequest]->
    kokkos_device = std::is_same<devicetype,lmpdevicetype>::value;

  if (neighflag == half || neighflag == halfthread) { // still need atomics, even though using a full neigh list
    neighbor->requests[irequest]->full = 1;
    neighbor->requests[irequest]->half = 0;
  } else {
    error->all(flerr,"must use half neighbor list style with pair snap/kk");
  }
}

/* ---------------------------------------------------------------------- */

template<class devicetype>
struct findmaxnumneighs {
  typedef devicetype device_type;
  neighlistkokkos<devicetype> k_list;

  findmaxnumneighs(neighlistkokkos<devicetype>* nl): k_list(*nl) {}
  ~findmaxnumneighs() {k_list.copymode = 1;}

  __attribute__((always_inline))
  void operator() (const int& ii, int& max_neighs) const {
    const int i = k_list.d_ilist[ii];
    const int num_neighs = k_list.d_numneigh[i];
    if (max_neighs<num_neighs) max_neighs = num_neighs;
  }
};

/* ----------------------------------------------------------------------
   this version is a straightforward implementation
   ---------------------------------------------------------------------- */

template<class typename real_type, int vector_length>
void pairsnapsycl<real_type, vector_length>::compute(int eflag_in, int vflag_in)
{
  if (host_flag) {
    atomkk->sync(host,x_mask|type_mask);
    pairsnap::compute(eflag_in,vflag_in);
    atomkk->modified(host,f_mask);
    return;
  }

  eflag = eflag_in;
  vflag = vflag_in;

  if (neighflag == full) no_virial_fdotr_compute = 1;

  ev_init(eflag,vflag,0);

  // reallocate per-atom arrays if necessary

  if (eflag_atom) {
    memorykk->destroy_kokkos(k_eatom,eatom);
    memorykk->create_kokkos(k_eatom,eatom,maxeatom,"pair:eatom");
    d_eatom = k_eatom.view<devicetype>();
  }
  if (vflag_atom) {
    memorykk->destroy_kokkos(k_vatom,vatom);
    memorykk->create_kokkos(k_vatom,vatom,maxvatom,"pair:vatom");
    d_vatom = k_vatom.view<devicetype>();
  }

  copymode = 1;
  int newton_pair = force->newton_pair;
  if (newton_pair == false)
    error->all(flerr,"pairsnapsycl requires 'newton on'");

  atomkk->sync(execution_space,x_mask|f_mask|type_mask);
  x = atomkk->k_x.view<devicetype>();
  f = atomkk->k_f.view<devicetype>();
  type = atomkk->k_type.view<devicetype>();
  k_cutsq.template sync<devicetype>();

  neighlistkokkos<devicetype>* k_list = static_cast<neighlistkokkos<devicetype>*>(list);
  d_numneigh = k_list->d_numneigh;
  d_neighbors = k_list->d_neighbors;
  d_ilist = k_list->d_ilist;
  inum = list->inum;

  need_dup = lmp->kokkos->need_dup<devicetype>();
  if (need_dup) {
    dup_f     = kokkos::experimental::create_scatter_view<kokkos::experimental::scattersum, kokkos::experimental::scatterduplicated>(f);
    dup_vatom = kokkos::experimental::create_scatter_view<kokkos::experimental::scattersum, kokkos::experimental::scatterduplicated>(d_vatom);
  } else {
    ndup_f     = kokkos::experimental::create_scatter_view<kokkos::experimental::scattersum, kokkos::experimental::scatternonduplicated>(f);
    ndup_vatom = kokkos::experimental::create_scatter_view<kokkos::experimental::scattersum, kokkos::experimental::scatternonduplicated>(d_vatom);
  }

  /*
  for (int i = 0; i < nlocal; i++) {
    typename t_neigh_list::t_neighs neighs_i = neigh_list.get_neighs(i);
    const int num_neighs = neighs_i.get_num_neighs();
    if (max_neighs<num_neighs) max_neighs = num_neighs;
  }*/
  max_neighs = 0;
  kokkos::parallel_reduce("pairsnapsycl::find_max_neighs",inum, findmaxnumneighs<devicetype>(k_list), kokkos::max<int>(max_neighs));

  int team_size_default = 1;
  if (!host_flag)
    team_size_default = 32;//max_neighs;

  if (beta_max < inum) {
    beta_max = inum;
    d_beta = sycl::malloc_device<real_type>(ncoeff * inum, squeue);
    //d_beta = kokkos::view<real_type**, devicetype>("pairsnapsycl:beta",ncoeff,inum);
    if (!host_flag)
      d_beta_pack = kokkos::view<real_type***, kokkos::layoutleft, devicetype>("pairsnapsycl:beta_pack",vector_length,ncoeff,(inum + vector_length - 1) / vector_length);
    d_ninside = kokkos::view<int*, devicetype>("pairsnapsycl:ninside",inum);
  }

  chunk_size = min(chunksize,inum); // "chunksize" variable is set by user
  chunk_offset = 0;

  snakk.grow_rij(chunk_size,max_neighs);

  ev_float ev;

  while (chunk_offset < inum) { // chunk up loop to prevent running out of memory

    ev_float ev_tmp;

    if (chunk_size > inum - chunk_offset)
      chunk_size = inum - chunk_offset;

    if (host_flag)
    {
      // ABB: removed because of KOKKOS CPU code path
    }
    else { // gpu

#ifdef lmp_kokkos_gpu

      // pre-compute ceil(chunk_size / vector_length) for code cleanliness
      const int chunk_size_div = (chunk_size + vector_length - 1) / vector_length;

      //computeneigh
      {
        // team_size_compute_neigh is defined in `pair_snap_kokkos.h`
        int scratch_size = scratch_size_helper<int>(team_size_compute_neigh * max_neighs);

        snapaosoateampolicy<team_size_compute_neigh, tagpairsnapcomputeneigh> policy_neigh(chunk_size,team_size_compute_neigh,vector_length);
        policy_neigh = policy_neigh.set_scratch_size(0, kokkos::perteam(scratch_size));
        kokkos::parallel_for("computeneigh",policy_neigh,*this);
      }

      //computecayleyklein
      {
        // tile_size_compute_ck is defined in `pair_snap_kokkos.h`
        snap3drangepolicy<tile_size_compute_ck, tagpairsnapcomputecayleyklein>
            policy_compute_ck({0,0,0},{vector_length,max_neighs,chunk_size_div},{vector_length,tile_size_compute_ck,1});
        kokkos::parallel_for("computecayleyklein",policy_compute_ck,*this);
      }

      //preui
      {
        // tile_size_pre_ui is defined in `pair_snap_kokkos.h`
        snap3drangepolicy<tile_size_pre_ui, tagpairsnappreui>
            policy_preui({0,0,0},{vector_length,twojmax+1,chunk_size_div},{vector_length,tile_size_pre_ui,1});
        kokkos::parallel_for("preui",policy_preui,*this);
      }

      // computeui w/vector parallelism, shared memory, direct atomicadd into ulisttot
      {
        // team_size_compute_ui is defined in `pair_snap_kokkos.h`
        // scratch size: 32 atoms * (twojmax+1) cached values, no double buffer
        const int tile_size = vector_length * (twojmax + 1);
        const int scratch_size = scratch_size_helper<complex>(team_size_compute_ui * tile_size);

        if (chunk_size < parallel_thresh)
        {
          // version with parallelism over j_bend

          // total number of teams needed: (natoms / 32) * (max_neighs) * ("bend" locations)
          const int n_teams = chunk_size_div * max_neighs * (twojmax + 1);
          const int n_teams_div = (n_teams + team_size_compute_ui - 1) / team_size_compute_ui;

          snapaosoateampolicy<team_size_compute_ui, tagpairsnapcomputeuismall> policy_ui(n_teams_div, team_size_compute_ui, vector_length);
          policy_ui = policy_ui.set_scratch_size(0, kokkos::perteam(scratch_size));
          kokkos::parallel_for("computeuismall",policy_ui,*this);
        } else {
          // version w/out parallelism over j_bend

          // total number of teams needed: (natoms / 32) * (max_neighs)
          const int n_teams = chunk_size_div * max_neighs;
          const int n_teams_div = (n_teams + team_size_compute_ui - 1) / team_size_compute_ui;

          snapaosoateampolicy<team_size_compute_ui, tagpairsnapcomputeuilarge> policy_ui(n_teams_div, team_size_compute_ui, vector_length);
          policy_ui = policy_ui.set_scratch_size(0, kokkos::perteam(scratch_size));
          kokkos::parallel_for("computeuilarge",policy_ui,*this);
        }
      }

      //transformui: un-"fold" ulisttot, zero ylist
      {
        // team_size_transform_ui is defined in `pair_snap_kokkos.h`
        snap3drangepolicy<tile_size_transform_ui, tagpairsnaptransformui>
            policy_transform_ui({0,0,0},{vector_length,snakk.idxu_max,chunk_size_div},{vector_length,tile_size_transform_ui,1});
        kokkos::parallel_for("transformui",policy_transform_ui,*this);
      }

      //compute bispectrum in aosoa data layout, transform bi
      if (quadraticflag || eflag) {
        // team_size_[compute_zi, compute_bi, transform_bi] are defined in `pair_snap_kokkos.h`

        //computezi
        const int idxz_max = snakk.idxz_max;
        snap3drangepolicy<tile_size_compute_zi, tagpairsnapcomputezi>
            policy_compute_zi({0,0,0},{vector_length,idxz_max,chunk_size_div},{vector_length,tile_size_compute_zi,1});
        kokkos::parallel_for("computezi",policy_compute_zi,*this);

        //computebi
        const int idxb_max = snakk.idxb_max;
        snap3drangepolicy<tile_size_compute_bi, tagpairsnapcomputebi>
            policy_compute_bi({0,0,0},{vector_length,idxb_max,chunk_size_div},{vector_length,tile_size_compute_bi,1});
        kokkos::parallel_for("computebi",policy_compute_bi,*this);

        //transform data layout of blist out of aosoa
        //we need this because `blist` gets used in computeforce which doesn't
        //take advantage of aosoa, which at best would only be beneficial on the margins
        snap3drangepolicy<tile_size_transform_bi, tagpairsnaptransformbi>
            policy_transform_bi({0,0,0},{vector_length,idxb_max,chunk_size_div},{vector_length,tile_size_transform_bi,1});
        kokkos::parallel_for("transformbi",policy_transform_bi,*this);
      }

      //note zeroing `ylist` is fused into `transformui`.
      {
        //compute beta = de_i/db_i for all i in list
        typename kokkos::rangepolicy<devicetype,tagpairsnapbeta> policy_beta(0,chunk_size);
        kokkos::parallel_for("computebeta",policy_beta,*this);
        const int idxz_max = snakk.idxz_max;
        if (quadraticflag || eflag) {
          snap3drangepolicy<tile_size_compute_yi, tagpairsnapcomputeyiwithzlist>
              policy_compute_yi({0,0,0},{vector_length,idxz_max,chunk_size_div},{vector_length,tile_size_compute_yi,1});
          kokkos::parallel_for("computeyiwithzlist",policy_compute_yi,*this);
        } else {
          snap3drangepolicy<tile_size_compute_yi, tagpairsnapcomputeyi>
              policy_compute_yi({0,0,0},{vector_length,idxz_max,chunk_size_div},{vector_length,tile_size_compute_yi,1});
          kokkos::parallel_for("computeyi",policy_compute_yi,*this);
        }
      }

      // fused computeduidrj, computedeidrj
      {
        // team_size_compute_fused_deidrj is defined in `pair_snap_kokkos.h`

        // scratch size: 32 atoms * (twojmax+1) cached values * 2 for u, du, no double buffer
        const int tile_size = vector_length * (twojmax + 1);
        const int scratch_size = scratch_size_helper<complex>(2 * team_size_compute_fused_deidrj * tile_size);

        if (chunk_size < parallel_thresh)
        {
          // version with parallelism over j_bend

          // total number of teams needed: (natoms / 32) * (max_neighs) * ("bend" locations)
          const int n_teams = chunk_size_div * max_neighs * (twojmax + 1);
          const int n_teams_div = (n_teams + team_size_compute_fused_deidrj - 1) / team_size_compute_fused_deidrj;

          // x direction
          snapaosoateampolicy<team_size_compute_fused_deidrj, tagpairsnapcomputefuseddeidrjsmall<0> > policy_fused_deidrj_x(n_teams_div,team_size_compute_fused_deidrj,vector_length);
          policy_fused_deidrj_x = policy_fused_deidrj_x.set_scratch_size(0, kokkos::perteam(scratch_size));
          kokkos::parallel_for("computefuseddeidrjsmall<0>",policy_fused_deidrj_x,*this);

          // y direction
          snapaosoateampolicy<team_size_compute_fused_deidrj, tagpairsnapcomputefuseddeidrjsmall<1> > policy_fused_deidrj_y(n_teams_div,team_size_compute_fused_deidrj,vector_length);
          policy_fused_deidrj_y = policy_fused_deidrj_y.set_scratch_size(0, kokkos::perteam(scratch_size));
          kokkos::parallel_for("computefuseddeidrjsmall<1>",policy_fused_deidrj_y,*this);

          // z direction
          snapaosoateampolicy<team_size_compute_fused_deidrj, tagpairsnapcomputefuseddeidrjsmall<2> > policy_fused_deidrj_z(n_teams_div,team_size_compute_fused_deidrj,vector_length);
          policy_fused_deidrj_z = policy_fused_deidrj_z.set_scratch_size(0, kokkos::perteam(scratch_size));
          kokkos::parallel_for("computefuseddeidrjsmall<2>",policy_fused_deidrj_z,*this);
        } else {
          // version w/out parallelism over j_bend

          // total number of teams needed: (natoms / 32) * (max_neighs)
          const int n_teams = chunk_size_div * max_neighs;
          const int n_teams_div = (n_teams + team_size_compute_fused_deidrj - 1) / team_size_compute_fused_deidrj;

          // x direction
          snapaosoateampolicy<team_size_compute_fused_deidrj, tagpairsnapcomputefuseddeidrjlarge<0> > policy_fused_deidrj_x(n_teams_div,team_size_compute_fused_deidrj,vector_length);
          policy_fused_deidrj_x = policy_fused_deidrj_x.set_scratch_size(0, kokkos::perteam(scratch_size));
          kokkos::parallel_for("computefuseddeidrjlarge<0>",policy_fused_deidrj_x,*this);

          // y direction
          snapaosoateampolicy<team_size_compute_fused_deidrj, tagpairsnapcomputefuseddeidrjlarge<1> > policy_fused_deidrj_y(n_teams_div,team_size_compute_fused_deidrj,vector_length);
          policy_fused_deidrj_y = policy_fused_deidrj_y.set_scratch_size(0, kokkos::perteam(scratch_size));
          kokkos::parallel_for("computefuseddeidrjlarge<1>",policy_fused_deidrj_y,*this);

          // z direction
          snapaosoateampolicy<team_size_compute_fused_deidrj, tagpairsnapcomputefuseddeidrjlarge<2> > policy_fused_deidrj_z(n_teams_div,team_size_compute_fused_deidrj,vector_length);
          policy_fused_deidrj_z = policy_fused_deidrj_z.set_scratch_size(0, kokkos::perteam(scratch_size));
          kokkos::parallel_for("computefuseddeidrjlarge<2>",policy_fused_deidrj_z,*this);

        }
      }

#endif // lmp_kokkos_gpu

    }

    //computeforce
    {
      if (evflag) {
        if (neighflag == half) {
          typename kokkos::rangepolicy<devicetype,tagpairsnapcomputeforce<half,1> > policy_force(0,chunk_size);
          kokkos::parallel_reduce(policy_force, *this, ev_tmp);
        } else if (neighflag == halfthread) {
          typename kokkos::rangepolicy<devicetype,tagpairsnapcomputeforce<halfthread,1> > policy_force(0,chunk_size);
          kokkos::parallel_reduce(policy_force, *this, ev_tmp);
        }
      } else {
        if (neighflag == half) {
          typename kokkos::rangepolicy<devicetype,tagpairsnapcomputeforce<half,0> > policy_force(0,chunk_size);
          kokkos::parallel_for(policy_force, *this);
        } else if (neighflag == halfthread) {
          typename kokkos::rangepolicy<devicetype,tagpairsnapcomputeforce<halfthread,0> > policy_force(0,chunk_size);
          kokkos::parallel_for(policy_force, *this);
        }
      }
    }
    ev += ev_tmp;
    chunk_offset += chunk_size;

  } // end while

  if (need_dup)
    kokkos::experimental::contribute(f, dup_f);

  if (eflag_global) eng_vdwl += ev.evdwl;
  if (vflag_global) {
    virial[0] += ev.v[0];
    virial[1] += ev.v[1];
    virial[2] += ev.v[2];
    virial[3] += ev.v[3];
    virial[4] += ev.v[4];
    virial[5] += ev.v[5];
  }

  if (vflag_fdotr) pair_virial_fdotr_compute(this);

  if (eflag_atom) {
    k_eatom.template modify<devicetype>();
    k_eatom.template sync<lmphosttype>();
  }

  if (vflag_atom) {
    if (need_dup)
      kokkos::experimental::contribute(d_vatom, dup_vatom);
    k_vatom.template modify<devicetype>();
    k_vatom.template sync<lmphosttype>();
  }

  atomkk->modified(execution_space,f_mask);

  copymode = 0;

  // free duplicated memory
  if (need_dup) {
    dup_f     = decltype(dup_f)();
    dup_vatom = decltype(dup_vatom)();
  }
}


/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

template<class typename real_type, int vector_length>
void pairsnapsycl<real_type, vector_length>::allocate()
{
  pairsnap::allocate();

  int n = atom->ntypes;
  d_map = kokkos::view<t_int*, devicetype>("pairsnapsycl::map",n+1);
}


/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

template<class typename real_type, int vector_length>
double pairsnapsycl<real_type, vector_length>::init_one(int i, int j)
{
  double cutone = pairsnap::init_one(i,j);
  k_cutsq.h_view(i,j) = k_cutsq.h_view(j,i) = cutone*cutone;
  k_cutsq.template modify<lmphosttype>();

  return cutone;
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairSNAP::coeff(int narg, char **arg)
{
  if (!allocated) allocate();
  if (narg != 4 + atom->ntypes) error->all(FLERR,"Incorrect args for pair coefficients");

  map_element2type(narg-4,arg+4);

  // read snapcoeff and snapparam files

  read_files(arg[2],arg[3]);

  if (!quadraticflag)
    ncoeff = ncoeffall - 1;
  else {

    // ncoeffall should be (ncoeff+2)*(ncoeff+1)/2
    // so, ncoeff = floor(sqrt(2*ncoeffall))-1

    ncoeff = sqrt(2*ncoeffall)-1;
    ncoeffq = (ncoeff*(ncoeff+1))/2;
    int ntmp = 1+ncoeff+ncoeffq;
    if (ntmp != ncoeffall) {
      error->all(FLERR,"Incorrect SNAP coeff file");
    }
  }

  snaptr = new SNA(lmp, rfac0, twojmax,
                   rmin0, switchflag, bzeroflag,
                   chemflag, bnormflag, wselfallflag, nelements);

  if (ncoeff != snaptr->ncoeff) {
    if (comm->me == 0)
      printf("ncoeff = %d snancoeff = %d \n",ncoeff,snaptr->ncoeff);
    error->all(FLERR,"Incorrect SNAP parameter file");
  }

  // Calculate maximum cutoff for all elements
  rcutmax = 0.0;
  for (int ielem = 0; ielem < nelements; ielem++)
    rcutmax = MAX(2.0*radelem[ielem]*rcutfac,rcutmax);

  // set default scaling
  int n = atom->ntypes;
  for (int ii = 0; ii < n+1; ii++)
    for (int jj = 0; jj < n+1; jj++)
      scale[ii][jj] = 1.0;

}

template<class typename real_type, int vector_length>
void pairsnapsycl<real_type, vector_length>::coeff(int narg, char **arg)
{
  pairsnap::coeff(narg,arg);

  // set up element lists

  d_radelem = kokkos::view<real_type*, devicetype>("pair:radelem",nelements);
  d_wjelem = kokkos::view<real_type*, devicetype>("pair:wjelem",nelements);
  d_coeffelem = kokkos::view<real_type**, kokkos::layoutright, devicetype>("pair:coeffelem",nelements,ncoeffall);

  auto h_radelem = kokkos::create_mirror_view(d_radelem);
  auto h_wjelem = kokkos::create_mirror_view(d_wjelem);
  auto h_coeffelem = kokkos::create_mirror_view(d_coeffelem);
  auto h_map = kokkos::create_mirror_view(d_map);

  for (int ielem = 0; ielem < nelements; ielem++) {
    h_radelem(ielem) = radelem[ielem];
    h_wjelem(ielem) = wjelem[ielem];
    for (int jcoeff = 0; jcoeff < ncoeffall; jcoeff++) {
      h_coeffelem(ielem,jcoeff) = coeffelem[ielem][jcoeff];
    }
  }

  for (int i = 1; i <= atom->ntypes; i++) {
    h_map(i) = map[i];
  }

  kokkos::deep_copy(d_radelem,h_radelem);
  kokkos::deep_copy(d_wjelem,h_wjelem);
  kokkos::deep_copy(d_coeffelem,h_coeffelem);
  kokkos::deep_copy(d_map,h_map);

  snakk = snakokkos<real_type, vector_length>(rfac0,twojmax,
                  rmin0,switchflag,bzeroflag,chemflag,bnormflag,wselfallflag,nelements);
  snakk.grow_rij(0,0);
  snakk.init();
}

/* ----------------------------------------------------------------------
   begin routines that are unique to the gpu codepath. these take advantage
   of aosoa data layouts and scratch memory for recursive polynomials
------------------------------------------------------------------------- */
template<class typename real_type, int vector_length>
__attribute__((always_inline))
void pairsnapsycl<real_type, vector_length>::tagpairsnapbeta() const {
  int i;
  int *type = atom->type;

  for (int ii = 0; ii < list->inum; ii++) {
    i = list->ilist[ii];
    const int itype = type[i];
    const int ielem = map[itype];
    double* coeffi = coeffelem[ielem];

    for (int icoeff = 0; icoeff < ncoeff; icoeff++)
      beta[ii][icoeff] = coeffi[icoeff+1];

    if (quadraticflag) {
      int k = ncoeff+1;
      for (int icoeff = 0; icoeff < ncoeff; icoeff++) {
        double bveci = bispectrum[ii][icoeff];
        beta[ii][icoeff] += coeffi[k]*bveci;
        k++;
        for (int jcoeff = icoeff+1; jcoeff < ncoeff; jcoeff++) {
          double bvecj = bispectrum[ii][jcoeff];
          beta[ii][icoeff] += coeffi[k]*bvecj;
          beta[ii][jcoeff] += coeffi[k]*bveci;
          k++;
        }
      }
    }
  }
}

template<class typename real_type, int vector_length>
__attribute__((always_inline))
void pairsnapsycl<real_type, vector_length>::operator() (tagpairsnapbeta,const int& ii) const {

  if (ii >= chunk_size) return;

  const int iatom_mod = ii % vector_length;
  const int iatom_div = ii / vector_length;

  const int i = d_ilist[ii + chunk_offset];
  const int itype = type[i];
  const int ielem = d_map[itype];
  snakokkos<real_type, vector_length> my_sna = snakk;

  auto d_coeffi = kokkos::subview(d_coeffelem, ielem, kokkos::all);

  for (int icoeff = 0; icoeff < ncoeff; icoeff++) {
    d_beta_pack(iatom_mod,icoeff,iatom_div) = d_coeffi[icoeff+1];
  }

  if (quadraticflag) {
    const auto idxb_max = my_sna.idxb_max;
    int k = ncoeff+1;
    for (int icoeff = 0; icoeff < ncoeff; icoeff++) {
      const auto idxb = icoeff % idxb_max;
      const auto idx_chem = icoeff / idxb_max;
      real_type bveci = my_sna.blist(ii, idx_chem, idxb);
      d_beta_pack(iatom_mod,icoeff,iatom_div) += d_coeffi[k]*bveci;
      k++;
      for (int jcoeff = icoeff+1; jcoeff < ncoeff; jcoeff++) {
        const auto jdxb = jcoeff % idxb_max;
        const auto jdx_chem = jcoeff / idxb_max;
        real_type bvecj = my_sna.blist(ii, jdx_chem, jdxb);
        d_beta_pack(iatom_mod,icoeff,iatom_div) += d_coeffi[k]*bvecj;
        d_beta_pack(iatom_mod,jcoeff,iatom_div) += d_coeffi[k]*bveci;
        k++;
      }
    }
  }
}

template<typename real_type, int vector_length>
__attribute__((always_inline))
void pairsnapsycl<real_type, vector_length>::tagpairsnapcomputeneigh() const {

  snakokkos<real_type, vector_length> my_sna = snakk;

  // extract atom number
  int ii = team.team_rank() + team.league_rank() * team.team_size();
  if (ii >= chunk_size) return;

  // get a pointer to scratch memory
  // this is used to cache whether or not an atom is within the cutoff.
  // if it is, type_cache is assigned to the atom type.
  // if it's not, it's assigned to -1.
  const int tile_size = max_neighs; // number of elements per thread
  const int team_rank = team.team_rank();
  const int scratch_shift = team_rank * tile_size; // offset into pointer for entire team
  int* type_cache = (int*)team.team_shmem().get_shmem(team.team_size() * tile_size * sizeof(int), 0) + scratch_shift;

  // shared memory
  using tile_t = int[16 * 64];
  tile_t& type_cache = *sycl::ext::oneapi::group_local_memory_for_overwrite<tile_t>(thread_block);

  // load various info about myself up front
  const int i = d_ilist[ii + chunk_offset];
  const f_float xtmp = x(i,0);
  const f_float ytmp = x(i,1);
  const f_float ztmp = x(i,2);
  const int itype = type[i];
  const int ielem = d_map[itype];
  const double radi = d_radelem[ielem];

  const int num_neighs = d_numneigh[i];

  // rij[][3] = displacements between atom i and those neighbors
  // inside = indices of neighbors of i within cutoff
  // wj = weights for neighbors of i within cutoff
  // rcutij = cutoffs for neighbors of i within cutoff
  // note rij sign convention => du/drij = du/drj = -du/dri

  // Compute the number of neighbors, store rsq
  int ninside = 0;
  Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team,num_neighs),
    [&] (const int jj, int& count) {
    T_INT j = d_neighbors(i,jj);
    const F_FLOAT dx = x(j,0) - xtmp;
    const F_FLOAT dy = x(j,1) - ytmp;
    const F_FLOAT dz = x(j,2) - ztmp;

    int jtype = type(j);
    const F_FLOAT rsq = dx*dx + dy*dy + dz*dz;

    if (rsq >= rnd_cutsq(itype,jtype)) {
      jtype = -1; // use -1 to signal it's outside the radius
    }

    type_cache[jj] = jtype;

    if (jtype >= 0)
     count++;
  }, ninside);

  d_ninside(ii) = ninside;

  sycl_queue->parallel_for<>(sycl::nd_range<>(globalsize, blockSize), [=](auto item) [[sycl::reqd_work_group_size(16)]] {
    const int jtype = type_cache[jj];

    if (jtype >= 0) {
      if (final) {
        T_INT j = d_neighbors(i,jj);
        const F_FLOAT dx = x(j,0) - xtmp;
        const F_FLOAT dy = x(j,1) - ytmp;
        const F_FLOAT dz = x(j,2) - ztmp;
        const int elem_j = d_map[jtype];
        my_sna.rij(ii,offset,0) = static_cast<real_type>(dx);
        my_sna.rij(ii,offset,1) = static_cast<real_type>(dy);
        my_sna.rij(ii,offset,2) = static_cast<real_type>(dz);
        my_sna.wj(ii,offset) = static_cast<real_type>(d_wjelem[elem_j]);
        my_sna.rcutij(ii,offset) = static_cast<real_type>((radi + d_radelem[elem_j])*rcutfac);
        my_sna.inside(ii,offset) = j;
        if (chemflag)
          my_sna.element(ii,offset) = elem_j;
        else
          my_sna.element(ii,offset) = 0;
      }
      offset++;
    }
  });
  
  Kokkos::parallel_scan(Kokkos::ThreadVectorRange(team,num_neighs),
    [&] (const int jj, int& offset, bool final) {

    const int jtype = type_cache[jj];

    if (jtype >= 0) {
      if (final) {
        T_INT j = d_neighbors(i,jj);
        const F_FLOAT dx = x(j,0) - xtmp;
        const F_FLOAT dy = x(j,1) - ytmp;
        const F_FLOAT dz = x(j,2) - ztmp;
        const int elem_j = d_map[jtype];
        my_sna.rij(ii,offset,0) = static_cast<real_type>(dx);
        my_sna.rij(ii,offset,1) = static_cast<real_type>(dy);
        my_sna.rij(ii,offset,2) = static_cast<real_type>(dz);
        my_sna.wj(ii,offset) = static_cast<real_type>(d_wjelem[elem_j]);
        my_sna.rcutij(ii,offset) = static_cast<real_type>((radi + d_radelem[elem_j])*rcutfac);
        my_sna.inside(ii,offset) = j;
        if (chemflag)
          my_sna.element(ii,offset) = elem_j;
        else
          my_sna.element(ii,offset) = 0;
      }
      offset++;
    }
  });
}

template<typename real_type, int vector_length>
__attribute__((always_inline))
void PairSNAPSYCL<real_type, vector_length>::operator() (TagPairSNAPComputeCayleyKlein,const int iatom_mod, const int jnbor, const int iatom_div) const {
  SNAKokkos<real_type, vector_length> my_sna = snaKK;

  const int ii = iatom_mod + iatom_div * vector_length;
  if (ii >= chunk_size) return;

  const int ninside = d_ninside(ii);
  if (jnbor >= ninside) return;

  my_sna.compute_cayley_klein(iatom_mod,jnbor,iatom_div);
}

template<typename real_type, int vector_length>
__attribute__((always_inline))
void PairSNAPSYCL<real_type, vector_length>::operator() (TagPairSNAPPreUi, const int iatom_mod, const int j, const int iatom_div) const {
  SNAKokkos<real_type, vector_length> my_sna = snaKK;

  const int ii = iatom_mod + iatom_div * vector_length;
  if (ii >= chunk_size) return;

  int itype = type(ii);
  int ielem = d_map[itype];

  my_sna.pre_ui(iatom_mod, j, ielem, iatom_div);
}

template<typename real_type, int vector_length>
__attribute__((always_inline))
void PairSNAPSYCL<real_type, vector_length>::operator() (TagPairSNAPComputeUiSmall,const typename Kokkos::TeamPolicy<DeviceType,TagPairSNAPComputeUiSmall>::member_type& team) const {
  SNAKokkos<real_type, vector_length> my_sna = snaKK;

  // extract flattened atom_div / neighbor number / bend location
  int flattened_idx = team.team_rank() + team.league_rank() * team_size_compute_ui;

  // extract neighbor index, iatom_div
  int iatom_div = flattened_idx / (max_neighs * (twojmax + 1)); // removed "const" to work around GCC 7 bug
  const int jj_jbend = flattened_idx - iatom_div * (max_neighs * (twojmax + 1));
  const int jbend = jj_jbend / max_neighs;
  int jj = jj_jbend - jbend * max_neighs; // removed "const" to work around GCC 7 bug

  Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, vector_length),
    [&] (const int iatom_mod) {
    const int ii = iatom_mod + vector_length * iatom_div;
    if (ii >= chunk_size) return;

    const int ninside = d_ninside(ii);
    if (jj >= ninside) return;

    my_sna.compute_ui_small(team, iatom_mod, jbend, jj, iatom_div);
  });

}

template<typename real_type, int vector_length>
__attribute__((always_inline))
void PairSNAPSYCL<real_type, vector_length>::operator() (TagPairSNAPComputeUiLarge,const typename Kokkos::TeamPolicy<DeviceType,TagPairSNAPComputeUiLarge>::member_type& team) const {
  SNAKokkos<real_type, vector_length> my_sna = snaKK;

  // extract flattened atom_div / neighbor number / bend location
  int flattened_idx = team.team_rank() + team.league_rank() * team_size_compute_ui;

  // extract neighbor index, iatom_div
  int iatom_div = flattened_idx / max_neighs; // removed "const" to work around GCC 7 bug
  int jj = flattened_idx - iatom_div * max_neighs;

  Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, vector_length),
    [&] (const int iatom_mod) {
    const int ii = iatom_mod + vector_length * iatom_div;
    if (ii >= chunk_size) return;

    const int ninside = d_ninside(ii);
    if (jj >= ninside) return;

    my_sna.compute_ui_large(team,iatom_mod, jj, iatom_div);
  });

}


template<typename real_type, int vector_length>
__attribute__((always_inline))
void PairSNAPSYCL<real_type, vector_length>::operator() (TagPairSNAPTransformUi,const int iatom_mod, const int idxu, const int iatom_div) const {
  SNAKokkos<real_type, vector_length> my_sna = snaKK;

  const int iatom = iatom_mod + iatom_div * vector_length;
  if (iatom >= chunk_size) return;

  if (idxu > my_sna.idxu_max) return;

  int elem_count = chemflag ? nelements : 1;

  for (int ielem = 0; ielem < elem_count; ielem++) {

    const FullHalfMapper mapper = my_sna.idxu_full_half[idxu];

    auto utot_re = my_sna.ulisttot_re_pack(iatom_mod, mapper.idxu_half, ielem, iatom_div);
    auto utot_im = my_sna.ulisttot_im_pack(iatom_mod, mapper.idxu_half, ielem, iatom_div);

    if (mapper.flip_sign == 1) {
      utot_im = -utot_im;
    } else if (mapper.flip_sign == -1) {
      utot_re = -utot_re;
    }

    my_sna.ulisttot_pack(iatom_mod, idxu, ielem, iatom_div) = { utot_re, utot_im };

    if (mapper.flip_sign == 0) {
      my_sna.ylist_pack_re(iatom_mod, mapper.idxu_half, ielem, iatom_div) = 0.;
      my_sna.ylist_pack_im(iatom_mod, mapper.idxu_half, ielem, iatom_div) = 0.;
    }
  }
}

template<typename real_type, int vector_length>
__attribute__((always_inline))
void PairSNAPSYCL<real_type, vector_length>::operator() (TagPairSNAPComputeYi,const int iatom_mod, const int jjz, const int iatom_div) const {
  SNAKokkos<real_type, vector_length> my_sna = snaKK;

  const int iatom = iatom_mod + iatom_div * vector_length;
  if (iatom >= chunk_size) return;

  if (jjz >= my_sna.idxz_max) return;

  my_sna.compute_yi(iatom_mod,jjz,iatom_div,d_beta_pack);
}

template<typename real_type, int vector_length>
__attribute__((always_inline))
void PairSNAPSYCL<real_type, vector_length>::operator() (TagPairSNAPComputeYiWithZlist,const int iatom_mod, const int jjz, const int iatom_div) const {
  SNAKokkos<real_type, vector_length> my_sna = snaKK;

  const int iatom = iatom_mod + iatom_div * vector_length;
  if (iatom >= chunk_size) return;

  if (jjz >= my_sna.idxz_max) return;

  my_sna.compute_yi_with_zlist(iatom_mod,jjz,iatom_div,d_beta_pack);
}

template<typename real_type, int vector_length>
__attribute__((always_inline))
void PairSNAPSYCL<real_type, vector_length>::operator() (TagPairSNAPComputeZi,const int iatom_mod, const int jjz, const int iatom_div) const {
  SNAKokkos<real_type, vector_length> my_sna = snaKK;

  const int iatom = iatom_mod + iatom_div * vector_length;
  if (iatom >= chunk_size) return;

  if (jjz >= my_sna.idxz_max) return;

  my_sna.compute_zi(iatom_mod,jjz,iatom_div);
}

template<typename real_type, int vector_length>
__attribute__((always_inline))
void PairSNAPSYCL<real_type, vector_length>::operator() (TagPairSNAPComputeBi,const int iatom_mod, const int jjb, const int iatom_div) const {
  SNAKokkos<real_type, vector_length> my_sna = snaKK;

  const int iatom = iatom_mod + iatom_div * vector_length;
  if (iatom >= chunk_size) return;

  if (jjb >= my_sna.idxb_max) return;

  my_sna.compute_bi(iatom_mod,jjb,iatom_div);
}

template<typename real_type, int vector_length>
__attribute__((always_inline))
void PairSNAPSYCL<real_type, vector_length>::operator() (TagPairSNAPTransformBi,const int iatom_mod, const int idxb, const int iatom_div) const {
  SNAKokkos<real_type, vector_length> my_sna = snaKK;

  const int iatom = iatom_mod + iatom_div * vector_length;
  if (iatom >= chunk_size) return;

  if (idxb >= my_sna.idxb_max) return;

  const int ntriples = my_sna.ntriples;

  for (int itriple = 0; itriple < ntriples; itriple++) {

    const real_type blocal = my_sna.blist_pack(iatom_mod, idxb, itriple, iatom_div);

    my_sna.blist(iatom, itriple, idxb) = blocal;
  }

}

template<typename real_type, int vector_length>
template<int dir>
__attribute__((always_inline))
void PairSNAPSYCL<real_type, vector_length>::operator() (TagPairSNAPComputeFusedDeidrjSmall<dir>,const typename Kokkos::TeamPolicy<DeviceType,TagPairSNAPComputeFusedDeidrjSmall<dir> >::member_type& team) const {
  SNAKokkos<real_type, vector_length> my_sna = snaKK;

  // extract flattened atom_div / neighbor number / bend location
  int flattened_idx = team.team_rank() + team.league_rank() * team_size_compute_fused_deidrj;

  // extract neighbor index, iatom_div
  int iatom_div = flattened_idx / (max_neighs * (twojmax + 1)); // removed "const" to work around GCC 7 bug
  const int jj_jbend = flattened_idx - iatom_div * (max_neighs * (twojmax + 1));
  const int jbend = jj_jbend / max_neighs;
  int jj = jj_jbend - jbend * max_neighs; // removed "const" to work around GCC 7 bug

  Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, vector_length),
    [&] (const int iatom_mod) {
    const int ii = iatom_mod + vector_length * iatom_div;
    if (ii >= chunk_size) return;

    const int ninside = d_ninside(ii);
    if (jj >= ninside) return;

    my_sna.template compute_fused_deidrj_small<dir>(team, iatom_mod, jbend, jj, iatom_div);

  });

}

template<typename real_type, int vector_length>
template<int dir>
__attribute__((always_inline))
void PairSNAPSYCL<real_type, vector_length>::operator() (TagPairSNAPComputeFusedDeidrjLarge<dir>,const typename Kokkos::TeamPolicy<DeviceType,TagPairSNAPComputeFusedDeidrjLarge<dir> >::member_type& team) const {
  SNAKokkos<real_type, vector_length> my_sna = snaKK;

  // extract flattened atom_div / neighbor number / bend location
  int flattened_idx = team.team_rank() + team.league_rank() * team_size_compute_fused_deidrj;

  // extract neighbor index, iatom_div
  int iatom_div = flattened_idx / max_neighs; // removed "const" to work around GCC 7 bug
  int jj = flattened_idx - max_neighs * iatom_div;

  Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, vector_length),
    [&] (const int iatom_mod) {
    const int ii = iatom_mod + vector_length * iatom_div;
    if (ii >= chunk_size) return;

    const int ninside = d_ninside(ii);
    if (jj >= ninside) return;

    my_sna.template compute_fused_deidrj_large<dir>(team, iatom_mod, jj, iatom_div);

  });
}

/* ----------------------------------------------------------------------
   Also used for both CPU and GPU codepaths. Could maybe benefit from a
   separate GPU/CPU codepath, but this kernel takes so little time it's
   likely not worth it.
------------------------------------------------------------------------- */

template<typename real_type, int vector_length>
template<int NEIGHFLAG, int EVFLAG>
__attribute__((always_inline))
void PairSNAPSYCL<real_type, vector_length>::operator() (TagPairSNAPComputeForce<NEIGHFLAG,EVFLAG>, const int& ii, EV_FLOAT& ev) const {

  // The f array is duplicated for OpenMP, atomic for CUDA, and neither for Serial
  auto v_f = ScatterViewHelper<typename NeedDup<NEIGHFLAG,DeviceType>::value,decltype(dup_f),decltype(ndup_f)>::get(dup_f,ndup_f);
  auto a_f = v_f.template access<typename AtomicDup<NEIGHFLAG,DeviceType>::value>();

  const int i = d_ilist[ii + chunk_offset];

  SNAKokkos<real_type, vector_length> my_sna = snaKK;

  const int ninside = d_ninside(ii);

  for (int jj = 0; jj < ninside; jj++) {
    int j = my_sna.inside(ii,jj);

    F_FLOAT fij[3];
    fij[0] = my_sna.dedr(ii,jj,0);
    fij[1] = my_sna.dedr(ii,jj,1);
    fij[2] = my_sna.dedr(ii,jj,2);

    a_f(i,0) += fij[0];
    a_f(i,1) += fij[1];
    a_f(i,2) += fij[2];
    a_f(j,0) -= fij[0];
    a_f(j,1) -= fij[1];
    a_f(j,2) -= fij[2];
    // tally global and per-atom virial contribution
    if (EVFLAG) {
      if (vflag_either) {
        v_tally_xyz<NEIGHFLAG>(ev,i,j,
          fij[0],fij[1],fij[2],
          -my_sna.rij(ii,jj,0),-my_sna.rij(ii,jj,1),
          -my_sna.rij(ii,jj,2));
      }
    }

  }
  // tally energy contribution

  if (EVFLAG) {
    if (eflag_either) {

      const int itype = type(i);
      const int ielem = d_map[itype];
      auto d_coeffi = Kokkos::subview(d_coeffelem, ielem, Kokkos::ALL);

      // evdwl = energy of atom I, sum over coeffs_k * Bi_k

      auto evdwl = d_coeffi[0];

      // E = beta.B + 0.5*B^t.alpha.B

      const auto idxb_max = snaKK.idxb_max;

      // linear contributions

      for (int icoeff = 0; icoeff < ncoeff; icoeff++) {
        const auto idxb = icoeff % idxb_max;
        const auto idx_chem = icoeff / idxb_max;
        evdwl += d_coeffi[icoeff+1]*my_sna.blist(ii,idx_chem,idxb);
      }

      // quadratic contributions
      if (quadraticflag) {
        int k = ncoeff+1;
        for (int icoeff = 0; icoeff < ncoeff; icoeff++) {
          const auto idxb = icoeff % idxb_max;
          const auto idx_chem = icoeff / idxb_max;
          real_type bveci = my_sna.blist(ii,idx_chem,idxb);
          evdwl += 0.5*d_coeffi[k++]*bveci*bveci;
          for (int jcoeff = icoeff+1; jcoeff < ncoeff; jcoeff++) {
            auto jdxb = jcoeff % idxb_max;
            auto jdx_chem = jcoeff / idxb_max;
            auto bvecj = my_sna.blist(ii,jdx_chem,jdxb);
            evdwl += d_coeffi[k++]*bveci*bvecj;
          }
        }
      }
      //ev_tally_full(i,2.0*evdwl,0.0,0.0,0.0,0.0,0.0);
      if (eflag_global) ev.evdwl += evdwl;
      if (eflag_atom) d_eatom[i] += evdwl;
    }
  }
}

template<typename real_type, int vector_length>
template<int NEIGHFLAG, int EVFLAG>
__attribute__((always_inline))
void PairSNAPSYCL<real_type, vector_length>::operator() (TagPairSNAPComputeForce<NEIGHFLAG,EVFLAG>,const int& ii) const {
  EV_FLOAT ev;
  this->template operator()<NEIGHFLAG,EVFLAG>(TagPairSNAPComputeForce<NEIGHFLAG,EVFLAG>(), ii, ev);
}

/* ---------------------------------------------------------------------- */

template<typename real_type, int vector_length>
template<int NEIGHFLAG>
__attribute__((always_inline))
void PairSNAPSYCL<real_type, vector_length>::v_tally_xyz(EV_FLOAT &ev, const int &i, const int &j,
								     const F_FLOAT &fx, const F_FLOAT &fy, const F_FLOAT &fz,
								     const F_FLOAT &delx, const F_FLOAT &dely, const F_FLOAT &delz) const
{
  // The vatom array is duplicated for OpenMP, atomic for CUDA, and neither for Serial

  auto v_vatom = ScatterViewHelper<typename NeedDup<NEIGHFLAG,DeviceType>::value,decltype(dup_vatom),decltype(ndup_vatom)>::get(dup_vatom,ndup_vatom);
  auto a_vatom = v_vatom.template access<typename AtomicDup<NEIGHFLAG,DeviceType>::value>();

  const E_FLOAT v0 = delx*fx;
  const E_FLOAT v1 = dely*fy;
  const E_FLOAT v2 = delz*fz;
  const E_FLOAT v3 = delx*fy;
  const E_FLOAT v4 = delx*fz;
  const E_FLOAT v5 = dely*fz;

  if (vflag_global) {
    ev.v[0] += v0;
    ev.v[1] += v1;
    ev.v[2] += v2;
    ev.v[3] += v3;
    ev.v[4] += v4;
    ev.v[5] += v5;
  }

  if (vflag_atom) {
    a_vatom(i,0) += 0.5*v0;
    a_vatom(i,1) += 0.5*v1;
    a_vatom(i,2) += 0.5*v2;
    a_vatom(i,3) += 0.5*v3;
    a_vatom(i,4) += 0.5*v4;
    a_vatom(i,5) += 0.5*v5;
    a_vatom(j,0) += 0.5*v0;
    a_vatom(j,1) += 0.5*v1;
    a_vatom(j,2) += 0.5*v2;
    a_vatom(j,3) += 0.5*v3;
    a_vatom(j,4) += 0.5*v4;
    a_vatom(j,5) += 0.5*v5;
  }
}

/* ----------------------------------------------------------------------
   memory usage
------------------------------------------------------------------------- */

template<typename real_type, int vector_length>
double PairSNAPSYCL<real_type, vector_length>::memory_usage()
{
  double bytes = Pair::memory_usage();
  int n = atom->ntypes+1;
  bytes += n*n*sizeof(int);
  bytes += n*n*sizeof(real_type);
  bytes += (2*ncoeffall)*sizeof(real_type);
  bytes += (ncoeff*3)*sizeof(real_type);
  bytes += snaKK.memory_usage();
  return bytes;
}

/* ---------------------------------------------------------------------- */

template<typename real_type, int vector_length>
template<class TagStyle>
void PairSNAPSYCL<real_type, vector_length>::check_team_size_for(int inum, int &team_size) {
  int team_size_max;

  team_size_max = Kokkos::TeamPolicy<DeviceType,TagStyle>(inum,Kokkos::AUTO).team_size_max(*this,Kokkos::ParallelForTag());

  if (team_size*vector_length > team_size_max)
    team_size = team_size_max/vector_length;
}

template<typename real_type, int vector_length>
template<class TagStyle>
void PairSNAPSYCL<real_type, vector_length>::check_team_size_reduce(int inum, int &team_size) {
  int team_size_max;

  team_size_max = Kokkos::TeamPolicy<DeviceType,TagStyle>(inum,Kokkos::AUTO).team_size_max(*this,Kokkos::ParallelReduceTag());

  if (team_size*vector_length > team_size_max)
    team_size = team_size_max/vector_length;
}

template<typename real_type, int vector_length>
template<typename scratch_type>
int PairSNAPSYCL<real_type, vector_length>::scratch_size_helper(int values_per_team) {
  typedef Kokkos::View<scratch_type*, Kokkos::DefaultExecutionSpace::scratch_memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged> > ScratchViewType;

  return ScratchViewType::shmem_size(values_per_team);
}



/* ----------------------------------------------------------------------
   routines used by template reference classes
------------------------------------------------------------------------- */

template<class DeviceType>
PairSNAPSYCLDevice<DeviceType>::PairSNAPSYCLDevice(class LAMMPS *lmp)
   : PairSNAPSYCL<SNAP_KOKKOS_REAL, SNAP_KOKKOS_DEVICE_VECLEN>(lmp) { ; }

template<class DeviceType>
void PairSNAPSYCLDevice<DeviceType>::coeff(int narg, char **arg)
{
  Base::coeff(narg, arg);
}

template<class DeviceType>
void PairSNAPSYCLDevice<DeviceType>::init_style()
{
  Base::init_style();
}

template<class DeviceType>
double PairSNAPSYCLDevice<DeviceType>::init_one(int i, int j)
{
  return Base::init_one(i, j);
}

template<class DeviceType>
void PairSNAPSYCLDevice<DeviceType>::compute(int eflag_in, int vflag_in)
{
  Base::compute(eflag_in, vflag_in);
}

template<class DeviceType>
double PairSNAPSYCLDevice<DeviceType>::memory_usage()
{
  return Base::memory_usage();
}

#ifdef LMP_KOKKOS_GPU
template<class DeviceType>
PairSNAPSYCLHost<DeviceType>::PairSNAPSYCLHost(class LAMMPS *lmp)
   : PairSNAPSYCL<SNAP_KOKKOS_REAL, SNAP_KOKKOS_HOST_VECLEN>(lmp) { ; }

template<class DeviceType>
void PairSNAPSYCLHost<DeviceType>::coeff(int narg, char **arg)
{
  Base::coeff(narg, arg);
}

template<class DeviceType>
void PairSNAPSYCLHost<DeviceType>::init_style()
{
  Base::init_style();
}

template<class DeviceType>
double PairSNAPSYCLHost<DeviceType>::init_one(int i, int j)
{
  return Base::init_one(i, j);
}

template<class DeviceType>
void PairSNAPSYCLHost<DeviceType>::compute(int eflag_in, int vflag_in)
{
  Base::compute(eflag_in, vflag_in);
}

template<class DeviceType>
double PairSNAPSYCLHost<DeviceType>::memory_usage()
{
  return Base::memory_usage();
}
#endif

}
