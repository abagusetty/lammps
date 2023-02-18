/* -----------------------------------------------------------------------
   Copyright (2010) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the Simplified BSD License.
   ----------------------------------------------------------------------- */

#ifndef SYCL_KERNEL
#define SYCL_KERNEL

#include "../lal_preprocessor.h"
#include "sycl_device.h"
#include <fstream>
#include <cstdio>

namespace ucl_sycl {

class UCL_Texture;
template <class numtyp> class UCL_D_Vec;
template <class numtyp> class UCL_D_Mat;
template <class hosttype, class devtype> class UCL_Vector;
template <class hosttype, class devtype> class UCL_Matrix;
#define UCL_MAX_KERNEL_ARGS 256

template <typename T, typename DIM>
using localAcc = sycl::accessor<T, DIM,
				sycl::access_mode::read_write,
				sycl::access::target::local>;

/// Class storing 1 or more kernel functions from a single string or file
class UCL_Program {
 public:
  inline UCL_Program() : _init_done(false) {}
  inline UCL_Program(UCL_Device &device) : _init_done(false) { init(device); }
  inline UCL_Program(UCL_Device &device, const void *program,
                     const char *flags="", std::string *log=nullptr) :
      _init_done(false) {
    init(device);
    load_string(program,flags,log);
  }

  inline ~UCL_Program() { clear(); }

  /// Initialize the program with a device
  inline void init(UCL_Device &device) {
    clear();

    _device=device.sycl_device();
    _context=device.context();
    _cq=device.cq();
    _init_done=true;
  }

  /// Clear any data associated with program
  /** \note Must call init() after each clear **/
  inline void clear() {
    if (_init_done) {
      _init_done=false;
    }
  }

  /// Load a program from a file and compile with flags
  inline int load(const char *filename, const char *flags="",
                  std::string *log=nullptr) {
    std::ifstream in(filename);
    if (!in || in.is_open()==false) {
      #ifndef UCL_NO_EXIT
      std::cerr << "UCL Error: Could not open kernel file: "
                << filename << std::endl;
      UCL_GERYON_EXIT;
      #endif
      return UCL_FILE_NOT_FOUND;
    }

    std::string program((std::istreambuf_iterator<char>(in)),
                        std::istreambuf_iterator<char>());
    in.close();
    return load_string(program.c_str(),flags,log);
  }

  /// Load a program from a string and compile with flags
  inline int load_string(const void *program, const char *flags="",
                         std::string *log=nullptr, FILE* foutput=nullptr) {
    // ABB: this is where may be I need to create the zeModule and stuff before the zeKernelCreate in set_function()
    auto ze_module_desc = ze_module_desc_t();
    ze_module_desc.stype = ZE_STRUCTURE_TYPE_MODULE_ZE_MODULE_DESC;
    ze_module_desc.pNext = nullptr;
    ze_module_desc.format = ZE_MODULE_FORMAT_NATIVE;
    ze_module_desc.inputSize = binary->size();
    ze_module_desc.pInputModule = binary->data();
    ze_module_desc.pBuildFlags = "-ze-opt-large-register-file";
    ze_module_desc.pConstants = nullptr;

    auto ze_device = sycl::get_native<ze_device_handle_t>(*_device);
    auto ze_ctxt = sycl::get_native<ze_context_handle_t>(*_context);

    ZE_SAFE_CALL( zeModuleCreate(ze_ctxt, ze_device, &ze_module_desc, &_ze_module, nullptr) );

    // sycl::make_kernel(sycl_kernel, kernel_name, sycl_engine, _ze_module,
    // 		      binary, programs);



    // cl_int error_flag;
    // const char *prog=(const char *)program;
    // _program=clCreateProgramWithSource(_context,1,&prog,nullptr,&error_flag);
    // CL_CHECK_ERR(error_flag);
    // error_flag = clBuildProgram(_program,1,&_device,flags,nullptr,nullptr);
    // if (error_flag!=-11)
    //   CL_CHECK_ERR(error_flag);
    // cl_build_status build_status;
    // clGetProgramBuildInfo(_program,_device,
    // 			  CL_PROGRAM_BUILD_STATUS,
    // 			  sizeof(cl_build_status),&build_status,
    // 			  nullptr);

    // if (build_status != CL_SUCCESS || log!=NULL) {
    //   size_t ms;
    //   clGetProgramBuildInfo(_program,_device,CL_PROGRAM_BUILD_LOG,0,NULL,&ms);
    //   char *build_log = new char[ms];
    //   clGetProgramBuildInfo(_program,_device,CL_PROGRAM_BUILD_LOG,ms,build_log, NULL);

    //   if (log!=nullptr)
    //     *log=std::string(build_log);

    //   if (build_status != CL_SUCCESS) {
    //     #ifndef UCL_NO_EXIT
    //     std::cerr << std::endl << std::endl
    //       << "----------------------------------------------------------\n"
    //       << " UCL Error: Error compiling OpenCL Program ("
    //       << build_status << ") ...\n"
    //       << "----------------------------------------------------------\n";
    //     std::cerr << build_log << std::endl;
    //     std::cerr <<
    //       "----------------------------------------------------------\n"
    //       << std::endl << std::endl;
    //     #endif
    //     if (foutput != NULL) {
    //       fprintf(foutput,"\n\n");
    //       fprintf(foutput,
    //         "----------------------------------------------------------\n");
    //       fprintf(foutput,
    //               " UCL Error: Error compiling OpenCL Program (%d) ...\n",
    //               build_status);
    //       fprintf(foutput,
    //         "----------------------------------------------------------\n");
    //       fprintf(foutput,"%s\n",build_log);
    //       fprintf(foutput,
    //         "----------------------------------------------------------\n");
    //       fprintf(foutput,"\n\n");
    //     }
    //     delete[] build_log;
    //     return UCL_COMPILE_ERROR;
    //   } else delete[] build_log;
    // }

    return UCL_SUCCESS;
  }

  /// Return the default SYCL queue/stream associated with this data
  inline command_queue* cq() { return _cq; }
  /// Change the default SYCL queue associated with matrix
  inline void cq(command_queue* cq_in) { _cq=cq_in; }

  friend class UCL_Kernel;
  friend class UCL_Const;
 private:
  bool _init_done;
  ze_module_handle_t _ze_module = nullptr;
  sycl::kernel_bundle _program;
  sycl::device* _device;
  sycl::context* _context;
  sycl::queue* _cq;
};

/// Class for dealing with SYCL kernels
class UCL_Kernel {
 public:
  UCL_Kernel() : _dimensions(1), _function_set(false), _num_args(0)
    {  _block_size[0]=0; _num_blocks[0]=0; }

  inline UCL_Kernel(UCL_Program &program, const char *function) :
    _dimensions(1), _function_set(false), _num_args(0)
    {  _block_size[0]=0; _num_blocks[0]=0; set_function(program,function); }

  inline ~UCL_Kernel() { clear(); }

  /// Clear any function associated with the kernel
  inline void clear() {
    if (_function_set) {
      clReleaseKernel(_kernel);
      clReleaseProgram(_program);
      clReleaseCommandQueue(_cq);
      _function_set=false;
    }
  }

  /// Get the kernel function from a program
  /** \return UCL_ERROR_FLAG (UCL_SUCCESS, UCL_FILE_NOT_FOUND, UCL_ERROR) **/
  inline int set_function(UCL_Program &program, const char *function);

  // /// Add a kernel argument.
  // template <class dtype>
  // inline void add_arg(const dtype * const arg) {
  //   clSetKernelArg(_kernel,_num_args,sizeof(dtype),arg);
  //   _num_args++;
  //   if (_num_args>UCL_MAX_KERNEL_ARGS) assert(0==1);
  // }

  // /// Add a geryon container as a kernel argument.
  // template <class numtyp>
  // inline void add_arg(const UCL_D_Vec<numtyp> * const arg)
  // { add_arg(&arg->begin()); }

  // /// Add a geryon container as a kernel argument.
  // template <class numtyp>
  // inline void add_arg(const UCL_D_Mat<numtyp> * const arg)
  // { add_arg(&arg->begin()); }

  // /// Add a geryon container as a kernel argument.
  // template <class hosttype, class devtype>
  // inline void add_arg(const UCL_Vector<hosttype, devtype> * const arg)
  // { add_arg(&arg->device.begin()); }

  // /// Add a geryon container as a kernel argument.
  // template <class hosttype, class devtype>
  // inline void add_arg(const UCL_Matrix<hosttype, devtype> * const arg)
  // { add_arg(&arg->device.begin()); }

  /// Set the number of thread blocks and the number of threads in each block
  /** \note This should be called before any arguments have been added
      \note The default SYCL queue is used for the kernel execution **/
  inline void set_size(const size_t num_blocks, const size_t block_size) {
    _dimensions=1;
    _num_blocks[0]=num_blocks*block_size;
    _block_size[0]=block_size;
  }

  /// Set the number of thread blocks and the number of threads in each block
  /** \note This should be called before any arguments have been added
      \note The default SYCL queue for the kernel is changed to cq **/
  inline void set_size(const size_t num_blocks, const size_t block_size,
                       command_queue* cq) {
    _cq=cq; set_size(num_blocks,block_size);
  }

  /// Set the number of thread blocks and the number of threads in each block
  /** \note This should be called before any arguments have been added
      \note The default SYCL queue is used for the kernel execution **/
  inline void set_size(const size_t num_blocks_x, const size_t num_blocks_y,
                       const size_t block_size_x, const size_t block_size_y) {
    _dimensions=2;
    _num_blocks[0]=num_blocks_x*block_size_x;
    _block_size[0]=block_size_x;
    _num_blocks[1]=num_blocks_y*block_size_y;
    _block_size[1]=block_size_y;
  }

  /// Set the number of thread blocks and the number of threads in each block
  /** \note This should be called before any arguments have been added
      \note The default SYCL queue for the kernel is changed to cq **/
  inline void set_size(const size_t num_blocks_x, const size_t num_blocks_y,
                       const size_t block_size_x, const size_t block_size_y,
                       command_queue* cq)
    {_cq=cq; set_size(num_blocks_x, num_blocks_y, block_size_x, block_size_y);}

  /// Set the number of thread blocks and the number of threads in each block
  /** \note This should be called before any arguments have been added
      \note The default SYCL queue is used for the kernel execution **/
  inline void set_size(const size_t num_blocks_x, const size_t num_blocks_y,
                       const size_t block_size_x,
                       const size_t block_size_y, const size_t block_size_z) {
    _dimensions=3;
    const size_t num_blocks_z=1;
    _num_blocks[0]=num_blocks_x*block_size_x;
    _block_size[0]=block_size_x;
    _num_blocks[1]=num_blocks_y*block_size_y;
    _block_size[1]=block_size_y;
    _num_blocks[2]=num_blocks_z*block_size_z;
    _block_size[2]=block_size_z;
  }

  /// Set the number of thread blocks and the number of threads in each block
  /** \note This should be called before any arguments have been added
      \note The default SYCL queue is used for the kernel execution **/
  inline void set_size(const size_t num_blocks_x, const size_t num_blocks_y,
                       const size_t block_size_x, const size_t block_size_y,
                       const size_t block_size_z, command_queue* cq) {
    _cq=cq;
    set_size(num_blocks_x, num_blocks_y, block_size_x, block_size_y,
             block_size_z);
  }

  /// Run the kernel in the default SYCL queue
  inline void run();

  /// Clear any arguments associated with the kernel
  inline void clear_args() { _num_args=0; }

  /// Return the default SYCL queue associated with this data
  inline command_queue* cq() { return _cq; }
  /// Change the default SYCL queue associated with matrix
  inline void cq(command_queue* cq_in) { _cq=cq_in; }

  #include "ucl_arg_kludge.h"

  inline size_t max_subgroup_size(const size_t block_size_x) {
    auto subg_sizes = _device->get_info<sycl::info::device::device::sub_group_sizes>();
    _mx_subgroup_sz = subg_sizes[0];
    return _mx_subgroup_sz;
  }
  inline size_t max_subgroup_size(const size_t block_size_x,
                                  const size_t block_size_y) {
    auto subg_sizes = _device->get_info<sycl::info::device::device::sub_group_sizes>();
    _mx_subgroup_sz = subg_sizes[0];
    return _mx_subgroup_sz;
  }
  inline size_t max_subgroup_size(const size_t block_size_x,
                                  const size_t block_size_y,
                                  const size_t block_size_z) {
    auto subg_sizes = _device->get_info<sycl::info::device::device::sub_group_sizes>();
    _mx_subgroup_sz = subg_sizes[0];
    return _mx_subgroup_sz;
  }

 private:
  sycl::kernel* _kernel;
  sycl::device* _device;
  uint _dimensions;
  size_t _block_size[3];
  size_t _num_blocks[3];
  bool _function_set;

  sycl::queue* _cq; // The default SYCL queue from UCL_Device for this kernel
  unsigned _num_args;

  size_t _mx_subgroup_sz;      // Maximum sub-group size for this kernel
};

inline int UCL_Kernel::set_function(UCL_Program &program, const char *function) {
  clear();

  _function_set=true;

  sycl::kernel_bundle<sycl::bundle_state::executable> kernel_bundle
    = sycl::make_kernel_bundle<sycl::backend::ext_oneapi_level_zero,
			       sycl::bundle_state::executable>({program._ze_module}, *(program._context));

  ze_kernel_handle_t ze_kernel = nullptr;

  ze_kernel_desc_t ze_kernel_desc = {};
  ze_kernel_desc.stype       = ZE_STRUCTURE_TYPE_KERNEL_DESC;
  ze_kernel_desc.pNext       = nullptr;
  ze_kernel_desc.flags       = 0;
  ze_kernel_desc.pKernelName = function;

  ZE_SAFE_CALL( zeKernelCreate(program._ze_module, &ze_kernel_desc, &ze_kernel) );

  auto k = sycl::make_kernel<sycl::backend::ext_oneapi_level_zero>
    ({kernel_bundle, ze_kernel}, *(program._context));
  //sycl_kernel = utils::make_unique<sycl::kernel>(k);


  _cq=program._cq;
  _program=program._program;
  _device=program._device;
  _kernel=clCreateKernel(_program,function,&error_flag);

  return UCL_SUCCESS;
}

template <typename... Args>
void UCL_Kernel::run(Args... args) {

}

void UCL_Kernel::run() {
  //clEnqueueNDRangeKernel(_cq,_kernel,_dimensions,NULL,_num_blocks,_block_size,0,NULL,NULL);

  // all I need here is the name of the function to call
  try {

    sycl::nd_range<_dimensions> execRange;
    if (_dimensions == 1) {
      execRange( sycl::range<1>(_num_blocks[0]),
		 sycl::range<1>(_block_size[0]) );
    } else if (_dimensions == 2) {
      execRange( sycl::range<2>(_num_blocks[0], _num_blocks[1]),
		 sycl::range<2>(_block_size[0], _block_size[1]) );
    } else if (_dimensions == 3) {
      execRange( sycl::range<3>(_num_blocks[0], _num_blocks[1], _num_blocks[2]),
		 sycl::range<3>(_block_size[0], _block_size[1], _block_size[2]) );
    }

    if (function_name == "calc_neigh_list_cell") {
      _cq->submit([&](sycl::handler& cgh) {
	  cgh.parallel_for(execRange, [=](sycl::nd_item<_dimensions> item) [[intel::reqd_sub_group_size(8)]] {
	      localAcc<int, 1> cell_list_sh_acc(BLOCK_NBOR_BUILD, cgh);
	      localAcc<numtyp4, 1> pos_sh_acc(BLOCK_NBOR_BUILD, cgh);

	      calc_neigh_list_cell( item, cell_list_sh_acc, pos_sh_acc, pos_tex,
				    args );
	    });
	});
    } else if ( function_name == "") {

    }

    if (_dimensions == 1) {
      sycl::nd_range<1> execRange( sycl::range<1>(_num_blocks[0]),
				   sycl::range<1>(_block_size[0]) );
      _cq->submit([&](sycl::handler& cgh) {
	  cgh.parallel_for(execRange, [=](sycl::nd_item<1> item)
			   [[intel::reqd_sub_group_size(8)]] {
			     _function( item, args );
	    });
	});
    }


    if (_dimensions == 1) {
      _cq->submit([&](sycl::handler& cgh) [[intel::reqd_sub_group_size(8)]] {
	  cgh.set_args(args);
	  cgh.parallel_for(execRange, *_kernel);
	});
    } else if (_dimensions == 2) {
      _cq->submit([&](sycl::handler& cgh) [[intel::reqd_sub_group_size(8)]] {
	  cgh.set_args(args);
	  cgh.parallel_for(execRange, *_kernel);
	});
    } else if (_dimensions == 3) {
      _cq->submit([&](sycl::handler& cgh) [[intel::reqd_sub_group_size(8)]] {
	  cgh.set_args(args);
	  cgh.parallel_for(execRange, *_kernel);
	});
    }

#ifdef GERYON_OCL_FLUSH
    ucl_flush(_cq);
#endif
  } catch ( sycl::exception const &e ) {
    std::cerr << "Sync exception: " << e.what() << std::endl;
    std::terminate();
  }

}

} // namespace

#endif
