// **************************************************************************
//                              neighbor_gpu.cpp
//                             -------------------
//                            Nitin Dhamankar (Intel)
//                              Peng Wang (Nvidia)
//                           W. Michael Brown (ORNL)
//
//  Device code for handling GPU generated neighbor lists
//
// __________________________________________________________________________
//    This file is part of the LAMMPS Accelerator Library (LAMMPS_AL)
// __________________________________________________________________________
//
//    begin                :
//    email                : penwang@nvidia.com, brownw@ornl.gov
// ***************************************************************************

#if defined(USE_SYCL)

#include "lal_preprocessor.h"

#ifdef LAMMPS_SMALLBIG
#define tagint int
#endif // LAMMPS_SMALLBIG

#ifdef LAMMPS_BIGBIG
#define tagint int64_t
#endif // LAMMPS_BIGBIG

#ifdef LAMMPS_SMALLSMALL
#define tagint int
#endif // LAMMPS_SMALLSMALL

#ifndef _DOUBLE_DOUBLE
dpct::image_wrapper<sycl::float4, 1> pos_tex;
#else
//ABB: may be, dpct::image_wrapper<sycl::int4, 2> pos_tex;
_texture_2d( pos_tex,int4);
#endif // _DOUBLE_DOUBLE

template <typename T, int DIM>
using localAcc = sycl::accessor<T, DIM,
				sycl::access_mode::read_write,
				sycl::access::target::local>;

void calc_cell_id(sycl::nd_item<3>& item,
		  dpct::image_accessor_ext<sycl::float4, 1> pos_tex,
		  const numtyp4 * x_,
		  unsigned * cell_id,
		  int * particle_id,
		  numtyp boxlo0, numtyp boxlo1, numtyp boxlo2,
		  numtyp i_cell_size, int ncellx, int ncelly,
		  int ncellz, int inum, int nall,
		  int cells_in_cutoff)
{
  int i = item.get_global_id(2);

  if (i < nall) {
    numtyp4 p;
    fetch4(p,i,pos_tex); //x_[i];

    p.x() -= boxlo0;
    p.y() -= boxlo1;
    p.z() -= boxlo2;

    int ix = int(p.x() * i_cell_size + cells_in_cutoff);
    int iy = int(p.y() * i_cell_size + cells_in_cutoff);
    int iz = int(p.z() * i_cell_size + cells_in_cutoff);

    int offset_lo, offset_hi;
    if (i<inum) {
      offset_lo=cells_in_cutoff;
      offset_hi=cells_in_cutoff+1;
    } else {
      offset_lo=0;
      offset_hi=1;
    }

    ix = sycl::max(ix, offset_lo);
    ix = sycl::min(ix, (int)(ncellx - offset_hi));
    iy = sycl::max(iy, offset_lo);
    iy = sycl::min(iy, (int)(ncelly - offset_hi));
    iz = sycl::max(iz, offset_lo);
    iz = sycl::min(iz, (int)(ncellz - offset_hi));

    cell_id[i] = ix+iy*ncellx+iz*ncellx*ncelly;
    particle_id[i] = i;
  }
}

void kernel_calc_cell_counts(const unsigned * cell_id,
			     int * cell_counts,
			     int nall, int ncell,
			     sycl::nd_item<3>& item) {
  int idx = item.get_global_id(2);
  if (idx < nall) {
    int id = cell_id[idx];

    // handle boundary cases
    if (idx == 0) {
      for (int i = 0; i < id + 1; i++)
        cell_counts[i] = 0;
    }
    if (idx == nall - 1) {
      for (int i = id+1; i <= ncell; i++)
        cell_counts[i] = nall;
    }

    if (idx > 0 && idx < nall) {
      int id_l = cell_id[idx-1];
      if (id != id_l) {
        for (int i = id_l+1; i <= id; i++)
          cell_counts[i] = idx;
      }
    }
  }
}

#else

#define pos_tex x_

#ifdef LAMMPS_SMALLBIG
#define tagint int
#endif

#ifdef LAMMPS_BIGBIG
#define tagint long
#endif

#ifdef LAMMPS_SMALLSMALL
#define tagint int
#endif

#endif // #if defined(USE_SYCL)

void transpose(tagint * out,
	       const tagint * in,
	       int columns_in, int rows_in,
	       sycl::nd_item<3>& item, // additional params for SYCL
	       localAcc<tagint, 2>& block)
{

  unsigned ti=THREAD_ID_X;
  unsigned tj=THREAD_ID_Y;
  unsigned bi=BLOCK_ID_X;
  unsigned bj=BLOCK_ID_Y;

  unsigned i=bi*BLOCK_CELL_2D+ti;
  unsigned j=bj*BLOCK_CELL_2D+tj;
  if ((i<columns_in) && (j<rows_in))
    block[tj][ti]=in[j*columns_in+i];

  item.barrier(sycl::access::fence_space::local_space);

  i=bj*BLOCK_CELL_2D+ti;
  j=bi*BLOCK_CELL_2D+tj;
  if ((i<rows_in) && (j<columns_in))
    out[j*rows_in+i] = block[ti][tj];
}

//Defines LAL_USE_OLD_NEIGHBOR for SYCL
void calc_neigh_list_cell(sycl::nd_item<3>& item, // additional params for SYCL
			  sycl::local_ptr<int> cell_list_sh,
			  sycl::local_ptr<numtype4> pos_sh,
			  dpct::image_accessor_ext<sycl::float4, 1>& pos_tex,
			  const sycl::device_ptr<numtyp4> x_,
			  const sycl::device_ptr<int> cell_particle_id,
			  const sycl::device_ptr<int> cell_counts,
			  sycl::device_ptr<int> nbor_list,
			  sycl::device_ptr<int> host_nbor_list,
			  sycl::device_ptr<int> host_numj,
			  int neigh_bin_size, numtyp cell_size,
			  int ncellx, int ncelly, int ncellz,
			  int inum, int nt, int nall, int t_per_atom,
			  int cells_in_cutoff)
{
  int tid = THREAD_ID_X;
  int ix = BLOCK_ID_X + cells_in_cutoff;
  int iy = BLOCK_ID_Y % (ncelly - cells_in_cutoff*2) + cells_in_cutoff;
  int iz = BLOCK_ID_Y / (ncelly - cells_in_cutoff*2) + cells_in_cutoff;
  int bsx = BLOCK_SIZE_X;

  int icell = ix + iy*ncellx + iz*ncellx*ncelly;

  int icell_begin = cell_counts[icell];
  int icell_end = cell_counts[icell+1];

  int nborz0 = iz-cells_in_cutoff, nborz1 = iz+cells_in_cutoff,
    nbory0 = iy-cells_in_cutoff, nbory1 = iy+cells_in_cutoff,
    nborx0 = ix-cells_in_cutoff, nborx1 = ix+cells_in_cutoff;

  numtyp4 diff;
  numtyp r2;
  int cap = sycl::ceil((numtyp)(icell_end - icell_begin) / bsx);
  for (int ii = 0; ii < cap; ii++) {
    int i = icell_begin + tid + ii*bsx;
    int pid_i = nall, pid_j, stride;
    numtyp4 atom_i, atom_j;
    int cnt = 0;
    int *neigh_counts, *neigh_list;

    if (i < icell_end)
      pid_i = cell_particle_id[i];

    if (pid_i < nt) {
      fetch4(atom_i,pid_i,pos_tex); //pos[i];
    }
    if (pid_i < inum) {
      stride=inum;
      neigh_counts=nbor_list+stride+pid_i;
      neigh_list=neigh_counts+stride+pid_i*(t_per_atom-1);
      stride=stride*t_per_atom-t_per_atom;
      nbor_list[pid_i]=pid_i;
    } else {
      stride=0;
      neigh_counts=host_numj+pid_i-inum;
      neigh_list=host_nbor_list+(pid_i-inum)*neigh_bin_size;
    }

    // loop through neighbors

    for (int nborz = nborz0; nborz <= nborz1; nborz++) {
      for (int nbory = nbory0; nbory <= nbory1; nbory++) {
        for (int nborx = nborx0; nborx <= nborx1; nborx++) {

          int jcell = nborx + nbory*ncellx + nborz*ncellx*ncelly;

          int jcell_begin = cell_counts[jcell];
          int jcell_end = cell_counts[jcell+1];
          int num_atom_cell = jcell_end - jcell_begin;

          // load jcell to shared memory
          int num_iter = sycl::ceil((numtyp)num_atom_cell / bsx);

          for (int k = 0; k < num_iter; k++) {
            int end_idx = sycl::min(bsx, (int)(num_atom_cell - k * bsx));

            if (tid < end_idx) {
              pid_j =  cell_particle_id[tid+k*bsx+jcell_begin];
              cell_list_sh[tid] = pid_j;
              fetch4(atom_j,pid_j,pos_tex); //[pid_j];
              pos_sh[tid].x() = atom_j.x();
              pos_sh[tid].y() = atom_j.y();
              pos_sh[tid].z() = atom_j.z();
            }
            item.barrier(sycl::access::fence_space::local_space);

            if (pid_i < nt) {

              for (int j = 0; j < end_idx; j++) {
                int pid_j = cell_list_sh[j]; // gather from shared memory
                diff.x() = atom_i.x() - pos_sh[j].x();
                diff.y() = atom_i.y() - pos_sh[j].y();
                diff.z() = atom_i.z() - pos_sh[j].z();

                r2 = diff.x() * diff.x() + diff.y() * diff.y() + diff.z() * diff.z();
                if (r2 < cell_size*cell_size && pid_j != pid_i) {
                  cnt++;
                  if (cnt <= neigh_bin_size) {
                    *neigh_list = pid_j;
                    neigh_list++;
                    if ((cnt & (t_per_atom-1))==0)
                      neigh_list=neigh_list+stride;
                  }
                }
              }
            }
            item.barrier(sycl::access::fence_space::local_space);
          } // for (k)
        }
      }
    }
    if (pid_i < nt)
      *neigh_counts = cnt;
  } // for (i)
}

void kernel_special(int *dev_nbor,
		    int *host_nbor_list,
		    const int *host_numj,
		    const tagint * tag,
		    const int * nspecial,
		    const tagint * special,
		    int inum, int nt, int max_nbors, int t_per_atom,
		    sycl::nd_item<3>& item) {

  int tid=THREAD_ID_X;
  int ii=fast_mul((int)BLOCK_ID_X,(int)(BLOCK_SIZE_X)/t_per_atom);
  ii += tid/t_per_atom;
  int offset=tid & (t_per_atom-1);

  if (ii<nt) {
    int stride;
    int *list, *list_end;

    int n1=nspecial[ii*3];
    int n2=nspecial[ii*3+1];
    int n3=nspecial[ii*3+2];

    int numj;
    if (ii < inum) {
      stride=inum;
      list=dev_nbor+stride+ii;
      numj=*list;
      list+=stride+fast_mul(ii,t_per_atom-1);
      stride=fast_mul(inum,t_per_atom);
      int njt=numj/t_per_atom;
      list_end=list+fast_mul(njt,stride)+(numj & (t_per_atom-1));
      list+=offset;
    } else {
      stride=1;
      list=host_nbor_list+(ii-inum)*max_nbors;
      numj=host_numj[ii-inum];
      list_end=list+fast_mul(numj,stride);
    }

    for ( ; list<list_end; list+=stride) {
      int nbor=*list;
      tagint jtag=tag[nbor];

      int offset=ii;
      for (int i=0; i<n3; i++) {
        if (special[offset]==jtag) {
          int which = 1;
          if (i>=n1)
            which++;
          if (i>=n2)
            which++;
          nbor=nbor ^ (which << SBBITS);
          *list=nbor;
        }
        offset+=nt;
      }
    }
  } // if ii
}
