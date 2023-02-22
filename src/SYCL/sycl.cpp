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

#include "sycl.h"

#include "error.h"
#include "force.h"
#include "memory_sycl.h"
#include "neigh_list_sycl.h"
#include "neighbor_sycl.h"

#include <cstring>
#include <cctype>
#include <csignal>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>            // for _getpid()
#else
#include <unistd.h>             // for getpid()
#endif

#ifdef LMP_SYCL_GPU

// for detecting GPU-aware MPI support:
// the variable int have_gpu_aware
// - is  1 if GPU-aware MPI support is available
// - is  0 if GPU-aware MPI support is unavailable
// - is -1 if GPU-aware MPI support is unknown

#define GPU_AWARE_UNKNOWN static int have_gpu_aware = -1;

// TODO HIP: implement HIP-aware MPI support (UCX) detection
#if defined(SYCL_ENABLE_HIP) || defined(SYCL_ENABLE_SYCL) || defined(SYCL_ENABLE_OPENMPTARGET)
GPU_AWARE_UNKNOWN
#elif defined(SYCL_ENABLE_CUDA)

// OpenMPI supports detecting GPU-aware MPI as of version 2.0.0

#if (OPEN_MPI)
#if (OMPI_MAJOR_VERSION >= 2)

#include <mpi-ext.h>

#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
static int have_gpu_aware = 1;
#elif defined(MPIX_CUDA_AWARE_SUPPORT) && !MPIX_CUDA_AWARE_SUPPORT
static int have_gpu_aware = 0;
#else
GPU_AWARE_UNKNOWN
#endif // defined(MPIX_CUDA_AWARE_SUPPORT)

#else // old OpenMPI
GPU_AWARE_UNKNOWN
#endif // (OMPI_MAJOR_VERSION >=2)

#else // unknown MPI library
GPU_AWARE_UNKNOWN
#endif // OPEN_MPI

#endif // SYCL_ENABLE_CUDA

#endif // LMP_ENABLE_DEVICE

using namespace LAMMPS_NS;

int SyclLMP::is_finalized = 0;
int SyclLMP::init_ngpus = 0;

/* ---------------------------------------------------------------------- */

SyclLMP::SyclLMP(LAMMPS *lmp, int narg, char **arg) : Pointers(lmp)
{
  sycl_exists = 1;
  lmp->sycl = this;

  exchange_comm_changed = 0;
  forward_comm_changed = 0;
  forward_pair_comm_changed = 0;
  reverse_pair_comm_changed = 0;
  forward_fix_comm_changed = 0;
  reverse_comm_changed = 0;

  delete memory;
  memory = new MemorySycl(lmp);
  memorySYCL = (MemorySycl*) memory;

  auto_sync = 1;
  allow_overlap = 1;

  int me = 0;
  MPI_Comm_rank(world,&me);
  if (me == 0) error->message(FLERR,"SYCL mode is enabled");

  // process any command-line args that invoke Sycl settings

  ngpus = 0;
  int device = 0;
  nthreads = 1;

  int iarg = 0;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"d") == 0 || strcmp(arg[iarg],"device") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Invalid Sycl command-line args");
      device = utils::inumeric(FLERR, arg[iarg+1], false, lmp);
      iarg += 2;

    } else if (strcmp(arg[iarg],"g") == 0 ||
               strcmp(arg[iarg],"gpus") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Invalid Sycl command-line args");
      ngpus = utils::inumeric(FLERR, arg[iarg+1], false, lmp);
#ifndef LMP_SYCL_GPU
      if (ngpus > 0)
        error->all(FLERR,"GPUs are requested but Sycl has not been compiled using a GPU-enabled backend");
#endif

      int skip_gpu = 9999;
      if (iarg+2 < narg && isdigit(arg[iarg+2][0])) {
        skip_gpu = utils::inumeric(FLERR, arg[iarg+2], false, lmp);
        iarg++;
      }
      iarg += 2;

      int set_flag = 0;
      char *str;
      if ((str = getenv("SLURM_LOCALID"))) {
        int local_rank = atoi(str);
        device = local_rank % ngpus;
        if (device >= skip_gpu) device++;
        set_flag = 1;
      }
      if ((str = getenv("MPT_LRANK"))) {
        if (ngpus > 0) {
          int local_rank = atoi(str);
          device = local_rank % ngpus;
          if (device >= skip_gpu) device++;
          set_flag = 1;
        }
      }
      if ((str = getenv("MV2_COMM_WORLD_LOCAL_RANK"))) {
        if (ngpus > 0) {
          int local_rank = atoi(str);
          device = local_rank % ngpus;
          if (device >= skip_gpu) device++;
          set_flag = 1;
        }
      }
      if ((str = getenv("OMPI_COMM_WORLD_LOCAL_RANK"))) {
        if (ngpus > 0) {
          int local_rank = atoi(str);
          device = local_rank % ngpus;
          if (device >= skip_gpu) device++;
          set_flag = 1;
        }
      }
      if ((str = getenv("PMI_LOCAL_RANK"))) {
        if (ngpus > 0) {
          int local_rank = atoi(str);
          device = local_rank % ngpus;
          if (device >= skip_gpu) device++;
          set_flag = 1;
        }
      }

      if (ngpus > 1 && !set_flag)
        error->all(FLERR,"Could not determine local MPI rank for multiple "
                           "GPUs with Sycl because MPI library not recognized");

    } else if (strcmp(arg[iarg],"t") == 0 ||
               strcmp(arg[iarg],"threads") == 0) {
      nthreads = utils::inumeric(FLERR, arg[iarg+1], false, lmp);

      if (nthreads <= 0)
        error->all(FLERR,"Invalid number of threads requested for Sycl: must be 1 or greater");

      iarg += 2;

    } else error->all(FLERR,"Invalid Sycl command-line arg: {}", arg[iarg]);
  }

  // Initialize Sycl. However, we cannot change any
  // Sycl library parameters after the first initalization

  Sycl::InitializationSettings args;

  if (args.has_num_threads()) {
    if ((args.get_num_threads() != nthreads) || (args.get_device_id() != device))
      if (me == 0)
        error->warning(FLERR,"Sycl package already initalized. Cannot change parameters");
    nthreads = args.get_num_threads();
    device = args.get_device_id();
    ngpus = init_ngpus;
  } else {
    args.set_num_threads(nthreads);
    args.set_device_id(device);
    init_ngpus = ngpus;
  }

  if ((me == 0) && (ngpus > 0))
    utils::logmesg(lmp, "  will use up to {} GPU(s) per node\n", ngpus);

#ifdef LMP_SYCL_GPU
  if (ngpus <= 0)
    error->all(FLERR,"Sycl has been compiled with GPU-enabled backend but no GPUs are requested");
#endif

#if !defined(SYCL_ENABLE_OPENMP) && !defined(SYCL_ENABLE_THREADS)
  if (nthreads > 1)
    error->all(FLERR,"Multiple CPU threads are requested but Sycl has not been compiled using a threading-enabled backend");
#endif

#ifndef SYCL_ENABLE_SERIAL
  if (nthreads == 1 && me == 0)
    error->warning(FLERR,"When using a single thread, the Sycl Serial backend "
                         "(i.e. Makefile.sycl_mpi_only) gives better performance "
                         "than the OpenMP backend");
#endif

  SyclLMP::initialize(args,error);

  // default settings for package sycl command

  binsize = 0.0;
#ifdef SYCL_ENABLE_CUDA
  // TODO HIP: implement HIP-aware MPI testing
  gpu_aware_flag = 1;
#else
  gpu_aware_flag = 0;
#endif
  neigh_thread = 0;
  neigh_thread_set = 0;
  neigh_transpose = 0;
  if (ngpus > 0) {
    neighflag = FULL;
    neighflag_qeq = FULL;
    newtonflag = 0;

    exchange_comm_classic = forward_comm_classic = reverse_comm_classic = 0;
    forward_pair_comm_classic = reverse_pair_comm_classic = forward_fix_comm_classic = 0;

    exchange_comm_on_host = forward_comm_on_host = reverse_comm_on_host = 0;
  } else {
    if (nthreads > 1) {
      neighflag = HALFTHREAD;
      neighflag_qeq = HALFTHREAD;
    } else {
      neighflag = HALF;
      neighflag_qeq = HALF;
    }
    newtonflag = 1;

    exchange_comm_classic = forward_comm_classic = reverse_comm_classic = 1;
    forward_pair_comm_classic = reverse_pair_comm_classic = forward_fix_comm_classic = 1;

    exchange_comm_on_host = forward_comm_on_host = reverse_comm_on_host = 0;
  }

#ifdef LMP_SYCL_GPU

  // check and warn about GPU-aware MPI availability when using multiple MPI tasks
  // change default only if we can safely detect that GPU-aware MPI is not available

  int nmpi = 0;
  MPI_Comm_size(world,&nmpi);
  if (nmpi > 1) {

#if defined(MPI_VERSION) && (MPI_VERSION > 2)
    // Check for IBM Spectrum MPI

    int len;
    char mpi_version[MPI_MAX_LIBRARY_VERSION_STRING];
    MPI_Get_library_version(mpi_version, &len);
    if (strstr(&mpi_version[0], "Spectrum") != nullptr) {
      gpu_aware_flag = 0;
      char* str;
      if ((str = getenv("OMPI_MCA_pml_pami_enable_cuda")))
        if ((strcmp(str,"1") == 0)) {
          have_gpu_aware = 1;
          gpu_aware_flag = 1;
        }

      if (!gpu_aware_flag)
        if (me == 0)
          error->warning(FLERR,"The Spectrum MPI '-gpu' flag is not set. Disabling GPU-aware MPI");
    }
#endif

    if (gpu_aware_flag == 1 && have_gpu_aware == 0) {
      if (me == 0)
        error->warning(FLERR,"Turning off GPU-aware MPI since it is not detected, "
                       "use '-pk sycl gpu/aware on' to override");
      gpu_aware_flag = 0;
    } else if (have_gpu_aware == -1) { // maybe we are dealing with MPICH, MVAPICH2 or some derivative?
    // MVAPICH2
#if defined(MPICH) && defined(MVAPICH2_VERSION)
      char* str;
      gpu_aware_flag = 0;
      if ((str = getenv("MV2_USE_CUDA")))
        if ((strcmp(str,"1") == 0))
          gpu_aware_flag = 1;

      if (!gpu_aware_flag)
        if (me == 0)
          error->warning(FLERR,"MVAPICH2 'MV2_USE_CUDA' environment variable is not set. Disabling GPU-aware MPI");
    // pure MPICH or some unsupported MPICH derivative
#elif defined(MPICH) && !defined(MVAPICH2_VERSION)
      char* str;
      gpu_aware_flag = 0;
      if ((str = getenv("MPICH_GPU_SUPPORT_ENABLED")))
        if ((strcmp(str,"1") == 0))
          gpu_aware_flag = 1;

      if (!gpu_aware_flag && me == 0)
        error->warning(FLERR,"Detected MPICH. Disabling GPU-aware MPI");
#else
  if (me == 0)
    error->warning(FLERR,"Sycl with GPU-enabled backend assumes GPU-aware MPI is available,"
                   " but cannot determine if this is the case\n         try"
                   " '-pk sycl gpu/aware off' if getting segmentation faults");

#endif
    } // if (-1 == have_gpu_aware)
  } // nmpi > 0
#endif // LMP_ENABLE_DEVICE

#ifdef KILL_SYCL_ON_SIGSEGV
  signal(SIGSEGV, my_signal_handler);
#endif
}

/* ---------------------------------------------------------------------- */

void SyclLMP::initialize(const Sycl::InitializationSettings& args, Error *error)
{
  if (!Sycl::is_initialized()) {
    if (is_finalized)
      error->all(FLERR,"Sycl package already finalized, cannot re-initialize");
    Sycl::initialize(args);
  }
}

/* ---------------------------------------------------------------------- */

void SyclLMP::finalize()
{
  if (Sycl::is_initialized() && !is_finalized)
    Sycl::finalize();
  is_finalized = 1;
}

/* ----------------------------------------------------------------------
   invoked by package sycl command
------------------------------------------------------------------------- */

void SyclLMP::accelerator(int narg, char **arg)
{
  int iarg = 0;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"neigh") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal package sycl command");
      if (strcmp(arg[iarg+1],"full") == 0) neighflag = FULL;
      else if (strcmp(arg[iarg+1],"half") == 0) {
        if (nthreads > 1 || ngpus > 0)
          neighflag = HALFTHREAD;
        else
          neighflag = HALF;
      }
      else error->all(FLERR,"Illegal package sycl command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"neigh/qeq") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal package sycl command");
      if (strcmp(arg[iarg+1],"full") == 0) neighflag_qeq = FULL;
      else if (strcmp(arg[iarg+1],"half") == 0) {
        if (nthreads > 1 || ngpus > 0)
          neighflag_qeq = HALFTHREAD;
        else
          neighflag_qeq = HALF;
      }
      else error->all(FLERR,"Illegal package sycl command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"binsize") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal package sycl command");
      binsize = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg],"newton") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal package sycl command");
      newtonflag = utils::logical(FLERR,arg[iarg+1],false,lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg],"comm") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal package sycl command");
      if (strcmp(arg[iarg+1],"no") == 0) {
        exchange_comm_classic = forward_comm_classic = reverse_comm_classic = 1;
        forward_pair_comm_classic = reverse_pair_comm_classic = forward_fix_comm_classic = 1;

        exchange_comm_on_host = forward_comm_on_host = reverse_comm_on_host = 0;
      } else if (strcmp(arg[iarg+1],"host") == 0) {
        exchange_comm_classic = forward_comm_classic = reverse_comm_classic = 0;
        forward_pair_comm_classic = reverse_pair_comm_classic = forward_fix_comm_classic = 1;

        exchange_comm_on_host = forward_comm_on_host = reverse_comm_on_host = 1;
      } else if (strcmp(arg[iarg+1],"device") == 0) {
        exchange_comm_classic = forward_comm_classic = reverse_comm_classic = 0;
        forward_pair_comm_classic = reverse_pair_comm_classic = forward_fix_comm_classic = 0;

        exchange_comm_on_host = forward_comm_on_host = reverse_comm_on_host = 0;
      } else error->all(FLERR,"Illegal package sycl command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"comm/exchange") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal package sycl command");
      if (strcmp(arg[iarg+1],"no") == 0) exchange_comm_classic = 1;
      else if (strcmp(arg[iarg+1],"host") == 0) {
        exchange_comm_classic = 0;
        exchange_comm_on_host = 1;
      } else if (strcmp(arg[iarg+1],"device") == 0) {
        exchange_comm_classic = 0;
        exchange_comm_on_host = 0;
      } else error->all(FLERR,"Illegal package sycl command");
      exchange_comm_changed = 0;
      iarg += 2;
    } else if (strcmp(arg[iarg],"comm/forward") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal package sycl command");
      if (strcmp(arg[iarg+1],"no") == 0) forward_comm_classic = 1;
      else if (strcmp(arg[iarg+1],"host") == 0) {
        forward_comm_classic = 0;
        forward_comm_on_host = 1;
      } else if (strcmp(arg[iarg+1],"device") == 0) {
        forward_comm_classic = 0;
        forward_comm_on_host = 0;
      } else error->all(FLERR,"Illegal package sycl command");
      forward_comm_changed = 0;
      iarg += 2;
    } else if (strcmp(arg[iarg],"comm/pair/forward") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal package sycl command");
      if (strcmp(arg[iarg+1],"no") == 0) forward_pair_comm_classic = 1;
      else if (strcmp(arg[iarg+1],"host") == 0) forward_pair_comm_classic = 1;
      else if (strcmp(arg[iarg+1],"device") == 0) forward_pair_comm_classic = 0;
      else error->all(FLERR,"Illegal package sycl command");
      forward_pair_comm_changed = 0;
      iarg += 2;
    } else if (strcmp(arg[iarg],"comm/pair/reverse") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal package sycl command");
      if (strcmp(arg[iarg+1],"no") == 0) reverse_pair_comm_classic = 1;
      else if (strcmp(arg[iarg+1],"host") == 0) reverse_pair_comm_classic = 1;
      else if (strcmp(arg[iarg+1],"device") == 0) reverse_pair_comm_classic = 0;
      else error->all(FLERR,"Illegal package sycl command");
      reverse_pair_comm_changed = 0;
      iarg += 2;
    } else if (strcmp(arg[iarg],"comm/fix/forward") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal package sycl command");
      if (strcmp(arg[iarg+1],"no") == 0) forward_fix_comm_classic = 1;
      else if (strcmp(arg[iarg+1],"host") == 0) forward_fix_comm_classic = 1;
      else if (strcmp(arg[iarg+1],"device") == 0) forward_fix_comm_classic = 0;
      else error->all(FLERR,"Illegal package sycl command");
      forward_fix_comm_changed = 0;
      iarg += 2;
    } else if (strcmp(arg[iarg],"comm/reverse") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal package sycl command");
      else if (strcmp(arg[iarg+1],"no") == 0) reverse_comm_classic = 1;
      else if (strcmp(arg[iarg+1],"host") == 0) {
        reverse_comm_classic = 0;
        reverse_comm_on_host = 1;
      } else if (strcmp(arg[iarg+1],"device") == 0) {
        reverse_comm_classic = 0;
        reverse_comm_on_host = 0;
      } else error->all(FLERR,"Illegal package sycl command");
      reverse_comm_changed = 0;
      iarg += 2;
    } else if ((strcmp(arg[iarg],"gpu/aware") == 0)
               || (strcmp(arg[iarg],"cuda/aware") == 0)) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal package sycl command");
      gpu_aware_flag = utils::logical(FLERR,arg[iarg+1],false,lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg],"pair/only") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal package sycl command");
      lmp->pair_only_flag = utils::logical(FLERR,arg[iarg+1],false,lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg],"neigh/thread") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal package sycl command");
      neigh_thread = utils::logical(FLERR,arg[iarg+1],false,lmp);
      neigh_thread_set = 1;
      iarg += 2;
    } else if (strcmp(arg[iarg],"neigh/transpose") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal package sycl command");
      neigh_transpose = utils::logical(FLERR,arg[iarg+1],false,lmp);
      iarg += 2;
    } else error->all(FLERR,"Illegal package sycl command");
  }

#ifdef LMP_SYCL_GPU

  int nmpi = 0;
  MPI_Comm_size(world,&nmpi);

  // if "gpu/aware off" or "pair/only on", and "comm device", change to "comm no"

  if ((!gpu_aware_flag && nmpi > 1) || lmp->pair_only_flag) {
    if (exchange_comm_classic == 0 && exchange_comm_on_host == 0) {
      exchange_comm_classic = 1;
      exchange_comm_changed = 1;
    }
    if (forward_comm_classic == 0 && forward_comm_on_host == 0) {
      forward_comm_classic = 1;
      forward_comm_changed = 1;
    }
    if (forward_pair_comm_classic == 0) {
      forward_pair_comm_classic = 1;
      forward_pair_comm_changed = 1;
    }
    if (reverse_pair_comm_classic == 0) {
      reverse_pair_comm_classic = 1;
      reverse_pair_comm_changed = 1;
    }
    if (forward_fix_comm_classic == 0) {
      forward_fix_comm_classic = 1;
      forward_fix_comm_changed = 1;
    }
    if (reverse_comm_classic == 0 && reverse_comm_on_host == 0) {
      reverse_comm_classic = 1;
      reverse_comm_changed = 1;
    }
  }

  // if "gpu/aware on" and "pair/only off", and comm flags were changed previously, change them back

  if (gpu_aware_flag && !lmp->pair_only_flag) {
    if (exchange_comm_changed) {
      exchange_comm_classic = 0;
      exchange_comm_changed = 0;
    }
    if (forward_comm_changed) {
      forward_comm_classic = 0;
      forward_comm_changed = 0;
    }
    if (forward_pair_comm_changed) {
      forward_pair_comm_classic = 0;
      forward_pair_comm_changed = 0;
    }
    if (reverse_pair_comm_changed) {
      reverse_pair_comm_classic = 0;
      reverse_pair_comm_changed = 0;
    }
    if (forward_fix_comm_changed) {
      forward_fix_comm_classic = 0;
      forward_fix_comm_changed = 0;
    }
    if (reverse_comm_changed) {
      reverse_comm_classic = 0;
      reverse_comm_changed = 0;
    }
  }

#endif

  // set newton flags
  // set neighbor binsize, same as neigh_modify command

  force->newton = force->newton_pair = force->newton_bond = newtonflag;

  if (neigh_thread && neighflag != FULL)
    error->all(FLERR,"Must use SYCL package option 'neigh full' with 'neigh/thread on'");

  neighbor->binsize_user = binsize;
  if (binsize <= 0.0) neighbor->binsizeflag = 0;
  else neighbor->binsizeflag = 1;
}

/* ----------------------------------------------------------------------
   called by Finish
------------------------------------------------------------------------- */

int SyclLMP::neigh_count(int m)
{
  int inum;
  int nneigh = 0;

  ArrayTypes<LMPHostType>::t_int_1d h_ilist;
  ArrayTypes<LMPHostType>::t_int_1d h_numneigh;

  NeighborSycl *nk = (NeighborSycl *) neighbor;
  if (nk->lists[m]->execution_space == Host) {
    NeighListSycl<LMPHostType>* nlistKK = (NeighListSycl<LMPHostType>*) nk->lists[m];
    inum = nlistKK->inum;
    h_ilist = Sycl::create_mirror_view(nlistKK->d_ilist);
    h_numneigh = Sycl::create_mirror_view(nlistKK->d_numneigh);
    Sycl::deep_copy(h_ilist,nlistKK->d_ilist);
    Sycl::deep_copy(h_numneigh,nlistKK->d_numneigh);
  } else if (nk->lists[m]->execution_space == Device) {
    NeighListSycl<LMPDeviceType>* nlistKK = (NeighListSycl<LMPDeviceType>*) nk->lists[m];
    inum = nlistKK->inum;
    h_ilist = Sycl::create_mirror_view(nlistKK->d_ilist);
    h_numneigh = Sycl::create_mirror_view(nlistKK->d_numneigh);
    Sycl::deep_copy(h_ilist,nlistKK->d_ilist);
    Sycl::deep_copy(h_numneigh,nlistKK->d_numneigh);
  }

  for (int i = 0; i < inum; i++) nneigh += h_numneigh[h_ilist[i]];

  return nneigh;
}

void SyclLMP::my_signal_handler(int sig)
{
  if (sig == SIGSEGV) {
#if defined(_WIN32)
    // there is no kill() function on Windows
    exit(1);
#else
    kill(getpid(),SIGABRT);
#endif
  }
}
