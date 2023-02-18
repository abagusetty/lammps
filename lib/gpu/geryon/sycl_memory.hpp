/***************************************************************************
                               sycl_memory.hpp
                             -------------------
                               W. Michael Brown

  OpenCL Specific Memory Management and Vector/Matrix Containers

 __________________________________________________________________________
    This file is part of the Geryon Unified Coprocessor Library (UCL)
 __________________________________________________________________________

    begin                : Wed Jan 13 2010
    copyright            : (C) 2010 by W. Michael Brown
    email                : brownw@ornl.gov
 ***************************************************************************/

/* -----------------------------------------------------------------------
   Copyright (2010) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the Simplified BSD License.
   ----------------------------------------------------------------------- */

#ifndef SYCL_MEMORY_HPP
#define SYCL_MEMORY_HPP

#include <iostream>
#include <cassert>
#include <cstring>
#include "ucl_types.h"

namespace ucl_sycl {

// --------------------------------------------------------------------------
// - API Specific Types
// --------------------------------------------------------------------------
struct ocl_kernel_dim {
  size_t x,y,z;
  ocl_kernel_dim(size_t _x = 1, size_t _y = 1, size_t _z = 1) :
    x(_x), y(_y), z(_z) {}
  operator size_t * () { return (size_t *)this; }
  operator const size_t * () const { return (const size_t *)this; }
};
typedef ocl_kernel_dim ucl_kernel_dim;

// --------------------------------------------------------------------------
// - API SPECIFIC DEVICE POINTERS
// --------------------------------------------------------------------------
typedef void* device_ptr;

// --------------------------------------------------------------------------
// - HOST MEMORY ALLOCATION ROUTINES
// --------------------------------------------------------------------------

template <class mat_type, class copy_type>
inline int _host_alloc(mat_type &mat, copy_type &cm, const size_t n,
                       const enum UCL_MEMOPT kind, const enum UCL_MEMOPT kind2){
  if (kind==UCL_NOT_PINNED)
    *(mat.host_ptr()) = static_cast<typename mat_type::data_type*>( malloc(n) );
  else
    *(mat.host_ptr()) = static_cast<typename mat_type::data_type*>( sycl::malloc_host(n, cm.cq()) );

  if (*(mat.host_ptr())==nullptr)
    return UCL_MEMORY_ERROR;

  mat.cq()=cm.cq();
  return UCL_SUCCESS;
}

template <class mat_type>
inline int _host_alloc(mat_type &mat, UCL_Device &dev, const size_t n,
                       const enum UCL_MEMOPT kind, const enum UCL_MEMOPT kind2){
  if (kind==UCL_NOT_PINNED)
    *(mat.host_ptr()) = static_cast<typename mat_type::data_type*>( malloc(n) );
  else
    *(mat.host_ptr()) = static_cast<typename mat_type::data_type*>( sycl::malloc_host(n, dev.cq()) );

  if (*(mat.host_ptr())==nullptr)
    return UCL_MEMORY_ERROR;

  mat.cq()=dev.cq();
  return UCL_SUCCESS;
}

template <class mat_type>
inline void _host_free(mat_type &mat) {
  if (mat.kind()==UCL_VIEW)
    return;
  else if (mat.kind()!=UCL_NOT_PINNED)
    sycl::free(mat.begin(), mat.cq());
  else
    free(mat.begin());
}

template <class mat_type>
inline int _host_resize(mat_type &mat, const size_t n) {
  _host_free(mat);
  
  if (mat.kind()==UCL_NOT_PINNED)
    *(mat.host_ptr())=static_cast<typename mat_type::data_type*>( malloc(n) );
  else
    *(mat.host_ptr())=static_cast<typename mat_type::data_type*>( sycl::malloc_host(n,mat.cq()) );

  if ( *(mat.host_ptr())==nullptr )
    return UCL_MEMORY_ERROR;
  return UCL_SUCCESS;
}

// --------------------------------------------------------------------------
// - DEVICE MEMORY ALLOCATION ROUTINES
// --------------------------------------------------------------------------

template <class mat_type, class copy_type>
inline int _device_alloc(mat_type &mat, copy_type &cm, const size_t n,
                         const enum UCL_MEMOPT kind) {
  mat.cbegin() = sycl::malloc_device(n, cm.cq());
  if (mat.cbegin() == nullptr)
    return UCL_MEMORY_ERROR;

  mat.cq()=cm.cq();
  return UCL_SUCCESS;
}

template <class mat_type>
inline int _device_alloc(mat_type &mat, UCL_Device &dev, const size_t n,
                         const enum UCL_MEMOPT kind) {
  mat.cbegin() = sycl::malloc_device(n, dev.cq());
  if (mat.cbegin() == nullptr)
    return UCL_MEMORY_ERROR;

  mat.cq()=dev.cq();
  return UCL_SUCCESS;
}

template <class mat_type, class copy_type>
inline int _device_alloc(mat_type &mat, copy_type &cm, const size_t rows,
                         const size_t cols, size_t &pitch,
                         const enum UCL_MEMOPT kind) {
  size_t padded_cols=cols;
  if (cols%256!=0)
    padded_cols+=256-cols%256;
  pitch=padded_cols*sizeof(typename mat_type::data_type);
  
  mat.cbegin() = sycl::aligned_alloc_device(16, pitch*rows, cm.cq());
  if (mat.cbegin == nullptr)
    return UCL_MEMORY_ERROR;
  mat.cq()=cm.cq();
  return UCL_SUCCESS;
}

template <class mat_type>
inline int _device_alloc(mat_type &mat, UCL_Device &dev, const size_t rows,
                         const size_t cols, size_t &pitch,
                         const enum UCL_MEMOPT kind) {
  size_t padded_cols=cols;
  if (dev.device_type()!=UCL_CPU && cols%256!=0)
    padded_cols+=256-cols%256;
  pitch=padded_cols*sizeof(typename mat_type::data_type);
  //return _device_alloc(mat,dev,pitch*rows,kind);

  mat.cbegin() = sycl::aligned_alloc_device(16, pitch*rows, dev.cq());
  if (mat.cbegin == nullptr)
    return UCL_MEMORY_ERROR;
  mat.cq()=dev.cq();
  return UCL_SUCCESS; 
}

template <class mat_type>
inline void _device_free(mat_type &mat) {
  if (mat.kind()!=UCL_VIEW)
    sycl::free(mat.cbegin(), mat.cq());
}

template <class mat_type>
inline int _device_resize(mat_type &mat, const size_t n) {
  _device_free(mat);

  mat.cbegin()=sycl::malloc_device(n, mat.cq());
  if (mat.cbegin()==nullptr)
    return UCL_MEMORY_ERROR;
}

template <class mat_type>
inline int _device_resize(mat_type &mat, const size_t rows,
                         const size_t cols, size_t &pitch) {
  size_t padded_cols=cols;
  if (cols%256!=0)
    padded_cols+=256-cols%256;
  pitch=padded_cols*sizeof(typename mat_type::data_type);

  mat.cbegin() = sycl::aligned_alloc_device(16, pitch*rows, mat.cq());
  if (mat.cbegin()==nullptr)
    return UCL_MEMORY_ERROR;

  return UCL_SUCCESS; 
}

inline void _device_view(device_ptr *ptr, device_ptr &in) {
  *ptr=in;
}

template <class numtyp>
inline void _device_view(device_ptr *ptr, numtyp *in) {
  *ptr=0;
}

inline void _device_view(device_ptr *ptr, device_ptr &in,
                         const size_t offset, const size_t numsize) {
  *ptr=in+offset*numsize;
}

template <class numtyp>
inline void _device_view(device_ptr *ptr, numtyp *in,
                         const size_t offset, const size_t numsize) {
  *ptr=0;
}

// --------------------------------------------------------------------------
// - ZERO ROUTINES
// --------------------------------------------------------------------------
inline void _host_zero(void *ptr, const size_t n) {
  memset(ptr,0,n);
}

inline void _ocl_build(cl_program &program, cl_device_id &device,
                       const char* options = "") {
  clBuildProgram(program,1,&device,options,nullptr,nullptr);

  cl_build_status build_status;
  CL_SAFE_CALL(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_STATUS,
                                     sizeof(cl_build_status),&build_status,
                                     nullptr));
  if (build_status == CL_SUCCESS)
    return;

  size_t ms;
  CL_SAFE_CALL(clGetProgramBuildInfo(program, device,CL_PROGRAM_BUILD_LOG, 0,
                                     nullptr, &ms));
  char *build_log = new char[ms];
  CL_SAFE_CALL(clGetProgramBuildInfo(program,device,CL_PROGRAM_BUILD_LOG,ms,
                                     build_log, nullptr));

  std::cerr << std::endl
            << "----------------------------------------------------------\n"
            << " Error compiling OpenCL Program...\n"
            << "----------------------------------------------------------\n";
  std::cerr << build_log << std::endl;
  delete[] build_log;
}

inline void _ocl_kernel_from_source(cl_context &context, cl_device_id &device,
                                    const char **source, const size_t lines,
                                    cl_kernel &kernel, const char *function,
                                    const char *options="") {
  cl_int error_flag;

  cl_program program=clCreateProgramWithSource(context,lines,source,
                                               nullptr,&error_flag);
  CL_CHECK_ERR(error_flag);
  _ocl_build(program,device,options);
  kernel=clCreateKernel(program,function,&error_flag);
  CL_CHECK_ERR(error_flag);
}

template <class mat_type>
inline void _device_zero(mat_type &mat, const size_t n, command_queue &cq) {
  cq.memset(mat.cbegin(), 0, n);

  #ifdef GERYON_OCL_FLUSH
  ucl_flush(cq);
  #endif
}

// --------------------------------------------------------------------------
// - MEMCPY ROUTINES
// --------------------------------------------------------------------------

template<int mem1, int mem2> struct _ucl_memcpy;

// Both are images
template<> struct _ucl_memcpy<2,2> {
  template <class p1, class p2>
  static inline void mc(p1 &dst, const p2 &src, const size_t n,
                        sycl::queue &cq, const bool block) {
    assert(0==1);
  }
  template <class p1, class p2>
  static inline void mc(p1 &dst, const size_t dpitch, const p2 &src,
                        const size_t spitch, const size_t cols,
                        const size_t rows, sycl::queue &cq,
                        const bool block) {
    assert(0==1);
  }
};

// Destination is texture, source on device
template<> struct _ucl_memcpy<2,0> {
  template <class p1, class p2>
  static inline void mc(p1 &dst, const p2 &src, const size_t n,
                        sycl::queue &cq, const bool block) {
    assert(0==1);
  }
  template <class p1, class p2>
  static inline void mc(p1 &dst, const size_t dpitch, const p2 &src,
                        const size_t spitch, const size_t cols,
                        const size_t rows, sycl::queue &cq,
                        const bool block) {
    assert(0==1);
  }
};

// Destination is texture, source on host
template<> struct _ucl_memcpy<2,1> {
  template <class p1, class p2>
  static inline void mc(p1 &dst, const p2 &src, const size_t n,
                        sycl::queue &cq, const bool block) {
    assert(0==1);
  }
  template <class p1, class p2>
  static inline void mc(p1 &dst, const size_t dpitch, const p2 &src,
                        const size_t spitch, const size_t cols,
                        const size_t rows, sycl::queue &cq,
                        const bool block) {
    assert(0==1);
  }
};

// Source is texture, dest on device
template<> struct _ucl_memcpy<0,2> {
  template <class p1, class p2>
  static inline void mc(p1 &dst, const p2 &src, const size_t n,
                        sycl::queue &cq, const bool block) {
    assert(0==1);
  }
  template <class p1, class p2>
  static inline void mc(p1 &dst, const size_t dpitch, const p2 &src,
                        const size_t spitch, const size_t cols,
                        const size_t rows, sycl::queue &cq,
                        const bool block) {
    assert(0==1);
  }
};

// Source is texture, dest on host
template<> struct _ucl_memcpy<1,2> {
  template <class p1, class p2>
  static inline void mc(p1 &dst, const p2 &src, const size_t n,
                        sycl::queue &cq, const bool block) {
    assert(0==1);
  }
  template <class p1, class p2>
  static inline void mc(p1 &dst, const size_t dpitch, const p2 &src,
                        const size_t spitch, const size_t cols,
                        const size_t rows, sycl::queue &cq,
                        const bool block) {
    assert(0==1);
  }
};

// Neither are textures, destination on host
template <> struct _ucl_memcpy<1,0> {
  template <class p1, class p2>
  static inline void mc(p1 &dst, const p2 &src, const size_t n,
                        sycl::queue &cq, const bool block) {
    cq.memcpy(dst.begin(), src.cbegin(), n);
    if (block) ucl_sync(cq);
    
    #ifdef GERYON_OCL_FLUSH
    if (!block) ucl_flush(cq);
    #endif
  }
  template <class p1, class p2>
  static inline void mc(p1 &dst, const size_t dpitch, const p2 &src,
                        const size_t spitch, const size_t cols,
                        const size_t rows, sycl::queue &cq, const bool block) {
    if (spitch==dpitch && dst.cols()==src.cols() &&
        src.cols()==cols/src.element_size()) {
      cq.memcpy(dst.begin(), src.cbegin(), spitch*rows);
    }
    else {
      for (size_t i=0; i<rows; i++) {
	
        CL_SAFE_CALL(clEnqueueReadBuffer(cq,src.cbegin(),block,src_offset,cols,
                                         (char *)dst.begin()+dst_offset,0,nullptr,
                                         nullptr));
        src_offset+=spitch;
        dst_offset+=dpitch;
      }
    }
    #ifdef GERYON_OCL_FLUSH
    if (!block) ucl_flush(cq);
    #endif
  }
};

// Neither are textures, source on host
template <> struct _ucl_memcpy<0,1> {
  template <class p1, class p2>
  static inline void mc(p1 &dst, const p2 &src, const size_t n,
                        sycl::queue &cq, const bool block,
                        const size_t dst_offset, const size_t src_offset) {
    if (src.cbegin()==dst.cbegin()) {
      if (block) ucl_sync(cq);
      return;
    }

    cq.memcpy(dst.cbegin()+dst_offset, src.begin()+src_offset, n);    
    if (block) ucl_sync(cq);
    
    #ifdef GERYON_OCL_FLUSH
    if (!block) ucl_flush(cq);
    #endif
  }
  template <class p1, class p2>
  static inline void mc(p1 &dst, const size_t dpitch, const p2 &src,
                        const size_t spitch, const size_t cols,
                        const size_t rows, sycl::queue &cq,
                        const bool block,
                        size_t dst_offset, size_t src_offset) {
    if (src.cbegin()==dst.cbegin()) {
      if (block) ucl_sync(cq);
      return;
    }
    
    if (spitch==dpitch && dst.cols()==src.cols() &&
        src.cols()==cols/src.element_size()) {
      cq.memcpy(dst.cbegin()+dst_offset, src.begin()+src_offset, spitch*rows);      
      if (block) ucl_sync(cq);
    }
    else {
      for (size_t i=0; i<rows; i++) {
	cq.memcpy(dst.cbegin()+dst_offset, src.begin()+src_offset, cols);      
	if (block) ucl_sync(cq);

        src_offset+=spitch;
        dst_offset+=dpitch;
      }
    }

#ifdef GERYON_OCL_FLUSH
    if (!block) ucl_flush(cq);
#endif
  }
};

// Neither are textures, both on device
template <int mem1, int mem2> struct _ucl_memcpy {
  template <class p1, class p2>
  static inline void mc(p1 &dst, const p2 &src, const size_t n,
                        sycl::queue &cq, const bool block,
                        const size_t dst_offset, const size_t src_offset) {
    if (src.cbegin()!=dst.cbegin() || src_offset!=dst_offset) {
      cq.memcpy(dst.cbegin()+dst_offset, src.cbegin()+src_offset, n);
    }

    if (block) ucl_sync(cq);
    #ifdef GERYON_OCL_FLUSH
    else ucl_flush(cq);
    #endif
  }
  template <class p1, class p2>
  static inline void mc(p1 &dst, const size_t dpitch, const p2 &src,
                        const size_t spitch, const size_t cols,
                        const size_t rows, sycl::queue &cq,
                        const bool block,
                        size_t dst_offset, size_t src_offset) {
    if (src.cbegin()!=dst.cbegin() || src_offset!=dst_offset) {
      if (spitch==dpitch && dst.cols()==src.cols() &&
          src.cols()==cols/src.element_size()) {
	cq.memcpy(dst.cbegin()+dst_offset, src.cbegin()+src_offset, spitch*rows);
      }
      else {
        for (size_t i=0; i<rows; i++) {
	  cq.memcpy(dst.cbegin()+dst_offset, src.cbegin()+src_offset, cols);
          src_offset+=spitch;
          dst_offset+=dpitch;
        }
      }
    }

    if (block) ucl_sync(cq);
    #ifdef GERYON_OCL_FLUSH
    else ucl_flush(cq);
    #endif
  }
};

template<class mat1, class mat2>
inline void ucl_mv_cpy(mat1 &dst, const mat2 &src, const size_t n) {
  _ucl_memcpy<mat1::MEM_TYPE,mat2::MEM_TYPE>::mc(dst,src,n,dst.cq(),true);
}

template<class mat1, class mat2>
inline void ucl_mv_cpy(mat1 &dst, const mat2 &src, const size_t n,
                       sycl::queue &cq) {
  _ucl_memcpy<mat1::MEM_TYPE,mat2::MEM_TYPE>::mc(dst,src,n,cq,false);
}

template<class mat1, class mat2>
inline void ucl_mv_cpy(mat1 &dst, const size_t dpitch, const mat2 &src,
                       const size_t spitch, const size_t cols,
                       const size_t rows) {
  _ucl_memcpy<mat1::MEM_TYPE,mat2::MEM_TYPE>::mc(dst,dpitch,src,spitch,cols,
                                                 rows,dst.cq(),true);
}

template<class mat1, class mat2>
inline void ucl_mv_cpy(mat1 &dst, const size_t dpitch, const mat2 &src,
                           const size_t spitch, const size_t cols,
                           const size_t rows,sycl::queue &cq) {
  _ucl_memcpy<mat1::MEM_TYPE,mat2::MEM_TYPE>::mc(dst,dpitch,src,spitch,cols,
                                                 rows,cq,false);
}

} // namespace ucl_sycl

#endif // SYCL_MEMORY_HPP
