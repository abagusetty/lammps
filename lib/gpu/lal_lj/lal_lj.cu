// **************************************************************************
//                                   lj.cu
//                             -------------------
//                           W. Michael Brown (ORNL)
//
//  Device code for acceleration of the lj/cut pair style
//
// __________________________________________________________________________
//    This file is part of the LAMMPS Accelerator Library (LAMMPS_AL)
// __________________________________________________________________________
//
//    begin                :
//    email                : brownw@ornl.gov
// ***************************************************************************

#include <cuda.h>
#include <cuda_runtime.h>

#include "lal_aux_fun1.h"

#if defined(NV_KERNEL) || defined(USE_HIP)
_texture( pos_tex,float4);
#else
#define pos_tex x_
#endif

#define fast_mul(X,Y) (X)*(Y)

#define fetch4(lans,i,pos_tex) {			\
    int4 xy = tex1Dfetch(pos_tex,i*2);		\
    int4 zt = tex1Dfetch(pos_tex,i*2+1);	\
    ans.x=__hiloint2double(xy.y, xy.x);		\
    ans.y=__hiloint2double(xy.w, xy.z);		\
    ans.z=__hiloint2double(zt.y, zt.x);		\
    ans.w=__hiloint2double(zt.w, zt.z);		\
  }
#define fetch(ans,i,q_tex) {			\
    int2 qt = tex1Dfetch(q_tex,i);		\
    ans=__hiloint2double(qt.y, qt.x);		\
  }

#define __kernel extern "C" __global__
#define __local __shared__
#define __global
#define atom_add atomicAdd
#define ucl_inline static __inline__ __device__

#define numtyp double
#define numtyp2 double2
#define numtyp4 double4
#define acctyp double
#define acctyp2 double2
#define acctyp4 double4

#define SBBITS 30
#define NEIGHMASK 0x3FFFFFFF
ucl_inline int sbmask(int j) { return j >> SBBITS & 3; };

#define MAX_SHARED_TYPES 8
#define restrict __restrict__
#define BLOCK_PAIR 256
#define SIMD_SIZE 32
#define simd_size() SIMD_SIZE

#define EVFLAG 1
#define NOUNROLL
#define GLOBAL_ID_X threadIdx.x+fast_mul(blockIdx.x,blockDim.x)
#define GLOBAL_ID_Y threadIdx.y+fast_mul(blockIdx.y,blockDim.y)
#define GLOBAL_SIZE_X fast_mul(gridDim.x,blockDim.x);
#define GLOBAL_SIZE_Y fast_mul(gridDim.y,blockDim.y);
#define THREAD_ID_X threadIdx.x
#define THREAD_ID_Y threadIdx.y
#define BLOCK_ID_X blockIdx.x
#define BLOCK_ID_Y blockIdx.y
#define BLOCK_SIZE_X blockDim.x
#define BLOCK_SIZE_Y blockDim.y
#define NUM_BLOCKS_X gridDim.x

#define simdsync() __syncwarp(0xffffffff)

__kernel void k_lj_fast(const __global numtyp4* restrict x_,
                        const __global numtyp4* restrict lj1_in,
                        const __global numtyp4* restrict lj3_in,
                        const __global numtyp* restrict sp_lj_in,
                        const __global int* dev_nbor,
                        const __global int* dev_packed,
                        __global acctyp4* restrict ans,
                        __global acctyp* restrict engv,
                        const int eflag, const int vflag, const int inum,
                        const int nbor_pitch, const int t_per_atom) {
  int tid, ii, offset;
  atom_info(t_per_atom,ii,tid,offset);

#ifndef ONETYPE
  __local numtyp4 lj1[MAX_SHARED_TYPES*MAX_SHARED_TYPES];
  __local numtyp4 lj3[MAX_SHARED_TYPES*MAX_SHARED_TYPES];
  __local numtyp sp_lj[4];
  if (tid<4)
    sp_lj[tid]=sp_lj_in[tid];
  if (tid<MAX_SHARED_TYPES*MAX_SHARED_TYPES) {
    lj1[tid]=lj1_in[tid];
    if (EVFLAG && eflag)
      lj3[tid]=lj3_in[tid];
  }
  __syncthreads();
  #else
  const numtyp lj1x=lj1_in[ONETYPE].x;
  const numtyp lj1y=lj1_in[ONETYPE].y;
  const numtyp cutsq=lj1_in[ONETYPE].z;
  numtyp lj3x, lj3y, lj3z;
  if (EVFLAG && eflag) {
    lj3x=lj3_in[ONETYPE].x;
    lj3y=lj3_in[ONETYPE].y;
    lj3z=lj3_in[ONETYPE].z;
  }
  #endif

  int n_stride;
  local_allocate_store_pair();

  acctyp4 f;
  f.x=(acctyp)0; f.y=(acctyp)0; f.z=(acctyp)0;
  acctyp energy, virial[6];
  if (EVFLAG) {
    energy=(acctyp)0;
    for (int i=0; i<6; i++) virial[i]=(acctyp)0;
  }

  if (ii<inum) {
    int i, numj, nbor, nbor_end;
    nbor_info(dev_nbor,dev_packed,nbor_pitch,t_per_atom,ii,offset,i,numj,
              n_stride,nbor_end,nbor);

    numtyp4 ix; fetch4(ix,i,pos_tex); //x_[i];
    #ifndef ONETYPE
    int iw=ix.w;
    int itype=fast_mul((int)MAX_SHARED_TYPES,iw);
    numtyp factor_lj;
    #endif

    NOUNROLL
    for ( ; nbor<nbor_end; nbor+=n_stride) {
      int j=dev_packed[nbor];
      #ifndef ONETYPE
      factor_lj = sp_lj[sbmask(j)];
      j &= NEIGHMASK;
      #endif

      numtyp4 jx; fetch4(jx,j,pos_tex); //x_[j];
      #ifndef ONETYPE
      int mtype=itype+jx.w;
      numtyp cutsq=lj1[mtype].z;
      #endif

      // Compute r12
      numtyp delx = ix.x-jx.x;
      numtyp dely = ix.y-jx.y;
      numtyp delz = ix.z-jx.z;
      numtyp r2inv = delx*delx+dely*dely+delz*delz;

      if (r2inv<cutsq) {
        #ifndef ONETYPE
        numtyp lj1x=lj1[mtype].x;
        numtyp lj1y=lj1[mtype].y;
        #endif

        r2inv=ucl_recip(r2inv);
        numtyp r6inv = r2inv*r2inv*r2inv;
        numtyp force = r2inv*r6inv*(lj1x*r6inv-lj1y);
        #ifndef ONETYPE
        force*=factor_lj;
        #endif

        f.x+=delx*force;
        f.y+=dely*force;
        f.z+=delz*force;

        if (EVFLAG && eflag) {
          #ifndef ONETYPE
          numtyp lj3x=lj3[mtype].x;
          numtyp lj3y=lj3[mtype].y;
          numtyp lj3z=lj3[mtype].z;
          #endif
          numtyp e=r6inv*(lj3x*r6inv-lj3y);
          #ifndef ONETYPE
          energy+=factor_lj*(e-lj3z);
          #else
          energy+=(e-lj3z);
          #endif
        }
        if (EVFLAG && vflag) {
          virial[0] += delx*delx*force;
          virial[1] += dely*dely*force;
          virial[2] += delz*delz*force;
          virial[3] += delx*dely*force;
          virial[4] += delx*delz*force;
          virial[5] += dely*delz*force;
        }
      }
    } // for nbor
  } // if ii
  store_answers(f,energy,virial,ii,inum,tid,t_per_atom,offset,eflag,vflag,
                ans,engv);
}

