/* -----------------------------------------------------------------------
   Copyright (2009) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the Simplified BSD License.
   ----------------------------------------------------------------------- */

#ifndef SYCL_DEVICE
#define SYCL_DEVICE

#include <string>
#include <vector>
#include <iostream>
#include <sycl/sycl.hpp>
#include "sycl_macros.hpp"
#include "ucl_types.h"

namespace ucl_sycl {

  // --------------------------------------------------------------------------
  // - SYCL ASYNCHRONOUS ERROR HANDLING
  // --------------------------------------------------------------------------
  auto sycl_asynchandler = [] (sycl::exception_list exceptions) {
    for (std::exception_ptr const& e : exceptions) {
      try {
        std::rethrow_exception(e);
      } catch (sycl::exception const& ex) {
        std::cout << "Caught asynchronous SYCL exception:" << std::endl
        << ex.what() << ", SYCL code: " << ex.code() << std::endl;
      }
    }
  };

  // --------------------------------------------------------------------------
  // - SYCL QUEUE STUFF
  // --------------------------------------------------------------------------
  typedef sycl::queue command_queue;

  inline void ucl_flush(command_queue& cq) { cq.wait_and_throw(); }

  inline void ucl_sync(command_queue& cq) { cq.wait(); }

#if defined(GERYON_FORCE_SHARED_MAIN_MEM_ON)
  inline bool _shared_mem_device(sycl::device& device) { return true; }
#elif defined(GERYON_FORCE_SHARED_MAIN_MEM_OFF)
  inline bool _shared_mem_device(sycl::device& device) { return false; }
#else
  inline bool _shared_mem_device(sycl::device& device) {
    return device.get_info<sycl::info::device::host_unified_memory>();
  }
#endif

  struct SYCLProperties {
    std::string name;
    sycl::info::device_type device_type;
    uint64_t global_mem;
    uint64_t shared_mem;
    uint32_t compute_units;
    uint32_t clock;
    size_t work_group_size;
    size_t work_item_size[3];
    bool has_double_precision;
    int preferred_vector_width32, preferred_vector_width64;
    int alignment;
    size_t timer_resolution;
    uint max_sub_devices;
    std::string sycl_device_version;
    bool has_subgroup_support;
  };

  /// Class for looking at data parallel device properties
  /** \note Calls to change the device outside of the class results in incorrect
   *       behavior
   * \note There is no error checking for indexing past the number of devices **/
  class UCL_Device {
  public:
    /// Collect properties for every device on the node
    /** \note You must set the active GPU with set() before using the device **/
    inline UCL_Device();

    inline ~UCL_Device();

    /// Return the number of platforms (0 if error or no platforms)
    inline int num_platforms() { return _num_platforms; }

    /// Return a string with name and info of the current platform
    inline std::string platform_name();

    /// Delete any contexts/data and set the platform number to be used
    inline int set_platform(const int pid);

    /// Return the number of devices that support SYCL
    inline int num_devices() { return _num_devices; }

    /// Set the SYCL device to the specified device number
    /** A context and default command queue will be created for the device *
     * Returns UCL_SUCCESS if successful or UCL_ERROR if the device could not
     * be allocated for use. clear() is called to delete any contexts and
     * associated data from previous calls to set(). **/
    inline int set(int num);

    /// Delete any context and associated data stored from a call to set()
    inline void clear();

    /// Get the current device number
    inline int device_num() { return _device; }

    /// Returns the context for the current device
    inline sycl::context* context() { return _context; }

    /// Returns the default queue for the current device
    inline command_queue* cq() { return cq(_default_cq); }

    /// Returns the queue indexed by i
    inline command_queue* cq(const int i) { return _cq[i]; }

    /// Set the default command queue
    /** \param i index of the command queue (as added by push_command_queue())
        If i is 0, the command queue created with device initialization is
        used **/
    inline void set_command_queue(const int i) { _default_cq=i; }

    /// Block until all commands in the default SYCL queue have completed
    inline void sync() { sync(_default_cq); }

    /// Block until all commands in the specified stream have completed
    inline void sync(const int i) { ucl_sync( *(cq(i)) ); }

    /// Get the number of SYCL queues currently available on device
    inline int num_queues() { return _cq.size(); }

    /// Add a SYCL queue for device computations (with profiling enabled)
    inline void push_command_queue() {
      _cq.push_back( new command_queue(*_context, *_sycl_device, sycl_asynchandler,
                                       sycl::property_list{sycl::property::queue::enable_profiling{},
                                           sycl::property::queue::in_order{}}) );

      if (_cq.back() == nullptr) {
        std::cerr << "Could not create SYCL queue on device: " << name()
                  << std::endl;
      }
    }

    /// Remove a stream for device computations
    /** \note You cannot delete the default stream **/
    inline void pop_command_queue() {
      if (_cq.size()<2) return;
      delete _cq.back();
      _cq.pop_back();
    }

    /// Get the current SYCL device name
    inline std::string name() { return name(_device); }
    /// Get the SYCL device name
    inline std::string name(const int i) {
      return std::string(_properties[i].name); }

    /// Get a string telling the type of the current device
    inline std::string device_type_name() { return device_type_name(_device); }
    /// Get a string telling the type of the device
    inline std::string device_type_name(const int i) { return "GPU"; };

    /// Get current device type (UCL_CPU, UCL_GPU, UCL_ACCELERATOR, UCL_DEFAULT)
    inline enum UCL_DEVICE_TYPE device_type() { return device_type(_device); }
    /// Get device type (UCL_CPU, UCL_GPU, UCL_ACCELERATOR, UCL_DEFAULT)
    inline enum UCL_DEVICE_TYPE device_type(const int i) { return UCL_GPU; };

    /// Returns true if host memory is efficiently addressable from device
    inline bool shared_memory() { return shared_memory(_device); }
    /// Returns true if host memory is efficiently addressable from device
    inline bool shared_memory(const int i)
    { return _shared_mem_device(_sycl_devices[i]); }

    /// Returns preferred vector width
    inline int preferred_fp32_width() { return preferred_fp32_width(_device); }
    /// Returns preferred vector width
    inline int preferred_fp32_width(const int i)
    {return _properties[i].preferred_vector_width32;}
    /// Returns preferred vector width
    inline int preferred_fp64_width() { return preferred_fp64_width(_device); }
    /// Returns preferred vector width
    inline int preferred_fp64_width(const int i)
    {return _properties[i].preferred_vector_width64;}

    /// Returns true if double precision is support for the current device
    inline bool double_precision() { return double_precision(_device); }
    /// Returns true if double precision is support for the device
    inline bool double_precision(const int i)
    {return _properties[i].has_double_precision;}

    /// Get the number of compute units on the current device
    inline unsigned cus() { return cus(_device); }
    /// Get the number of compute units
    inline unsigned cus(const int i)
    { return _properties[i].compute_units; }

    /// Get the gigabytes of global memory in the current device
    inline double gigabytes() { return gigabytes(_device); }
    /// Get the gigabytes of global memory
    inline double gigabytes(const int i)
    { return static_cast<double>(_properties[i].global_mem)/1073741824; }

    /// Get the bytes of global memory in the current device
    inline size_t bytes() { return bytes(_device); }
    /// Get the bytes of global memory
    inline size_t bytes(const int i) { return _properties[i].global_mem; }

    /// Return the GPGPU revision number for current device
    //inline double revision() { return revision(_device); }
    /// Return the GPGPU revision number
    //inline double revision(const int i)
    //  { return //static_cast<double>(_properties[i].minor)/10+_properties[i].major;}

    /// Clock rate in GHz for current device
    inline double clock_rate() { return clock_rate(_device); }
    /// Clock rate in GHz
    inline double clock_rate(const int i) { return _properties[i].clock*1e-3;}

    /// Return the address alignment in bytes
    inline int alignment() { return alignment(_device); }
    /// Return the address alignment in bytes
    inline int alignment(const int i) { return _properties[i].alignment; }

    /// Return the timer resolution
    inline size_t timer_resolution() { return timer_resolution(_device); }
    /// Return the timer resolution
    inline size_t timer_resolution(const int i)
    { return _properties[i].timer_resolution; }

    /// Get the maximum number of threads per block
    inline size_t group_size() { return group_size(_device); }
    /// Get the maximum number of threads per block
    inline size_t group_size(const int i)
    { return _properties[i].work_group_size; }
    /// Get the maximum number of threads per block in dimension 'dim'
    inline size_t group_size_dim(const int dim)
    { return group_size_dim(_device, dim); }
    /// Get the maximum number of threads per block in dimension 'dim'
    inline size_t group_size_dim(const int i, const int dim)
    { return _properties[i].work_item_size[dim]; }

    /// Get the shared local memory size in bytes
    inline size_t slm_size() { return slm_size(_device); }
    /// Get the shared local memory size in bytes
    inline size_t slm_size(const int i)
    { return _properties[i].shared_mem; }

    /// Return the maximum memory pitch in bytes for current device
    inline size_t max_pitch() { return max_pitch(_device); }
    /// Return the maximum memory pitch in bytes
    inline size_t max_pitch(const int i) { return 0; }

    /// Returns false if accelerator cannot be shared by multiple processes
    /** If it cannot be determined, true is returned **/
    inline bool sharing_supported() { return sharing_supported(_device); }
    /// Returns false if accelerator cannot be shared by multiple processes
    /** If it cannot be determined, true is returned **/
    inline bool sharing_supported(const int i) { return true; }

    /// True if the device has subgroup support
    inline bool has_subgroup_support()
    { return has_subgroup_support(_device); }
    /// True if the device has subgroup support
    inline bool has_subgroup_support(const int i)
    { return _properties[i].has_subgroup_support; }

    /// Maximum number of subdevices allowed from device fission
    inline int max_sub_devices() { return max_sub_devices(_device); }
    /// Maximum number of subdevices allowed from device fission
    inline int max_sub_devices(const int i) { return _properties[i].max_sub_devices; }

    /// List all devices along with all properties
    inline void print_all(std::ostream &out);

    /// Return the SYCL device
    inline sycl::device* sycl_device() { return _sycl_device; }

    /// Automatically set the platform by type, vendor, and/or CU count
    /** If first_device is positive, search restricted to platforms containing
     * this device IDs. If ndevices is positive, search is restricted
     * to platforms with at least that many devices  **/
    inline int auto_set_platform(const enum UCL_DEVICE_TYPE type=UCL_GPU,
                                 const std::string vendor="",
                                 const int ndevices=-1,
                                 const int first_device=-1);

  private:
    int _num_platforms;                          // Number of platforms
    int _platform;                               // UCL_Device ID for current platform
    sycl::platform _sycl_platform;               // SYCL ID for current platform
    std::vector<sycl::platform> _sycl_platforms; // SYCL IDs for all platforms
    sycl::context* _context;                     // Context used for accessing the device
    std::vector<command_queue*> _cq;             // The default command queue for this device
    int _device;                                 // UCL_Device ID for current device
    sycl::device* _sycl_device;                  // SYCL ID for current device
    std::vector<sycl::device> _sycl_devices;    // SYCL IDs for all devices
    int _num_devices;                            // Number of devices
    std::vector<SYCLProperties> _properties;     // Properties for each device

    inline void add_properties(sycl::device& dev);
    inline int create_context();
    int _default_cq;
  };

  // Grabs the properties for all devices
  UCL_Device::UCL_Device() {
    _device=-1;

    // --- Get Number of Platforms
    int nplatforms = 0;
    _sycl_platforms = std::move( sycl::platform::get_platforms() );
    nplatforms = _sycl_platforms.size();

    if (nplatforms == 0) {
      _num_platforms=0;
      return;
    } else {
      _num_platforms=nplatforms;
    }

    set_platform(0);
  }

  UCL_Device::~UCL_Device() {
    clear();
  }

  void UCL_Device::clear() {
    _properties.clear();

    _sycl_devices.clear();

    if (_device>-1) {
      for (size_t i=0; i<_cq.size(); i++) {
        delete _cq.back();
        _cq.pop_back();
      }
      delete _context;
    }
    _cq.clear();

    _device=-1;
    _num_devices=0;
  }

  int UCL_Device::set_platform(int pid) {
    clear();

    _sycl_device=nullptr;
    _device=-1;
    _num_devices=0;
    _default_cq=0;

#ifdef UCL_DEBUG
    assert(pid<num_platforms());
#endif
    _platform=pid;
    _sycl_platform=_sycl_platforms[_platform];

    // --- Get Number of Devices
    auto allDevices = _sycl_platform.get_devices(sycl::info::device_type::gpu);
    _num_devices=allDevices.size();
    if (_num_devices == 0) {
      _num_devices=0;
      return UCL_ERROR;
    }

#ifndef GERYON_NUMA_FISSION
    // --- Store properties for each device
    _sycl_devices = std::move(allDevices);
    for (int i=0; i<_num_devices; i++) {
      add_properties( _sycl_devices[i] );
    }
#else
    // --- Create sub-devices for anything partitionable by NUMA and store props
    int num_unpart = _num_devices;
    _num_devices = 0;
    for (int i=0; i<num_unpart; i++) {
      if (allDevices[i].get_info<sycl::info::device::partition_max_sub_devices>() > 1) {
	auto subdevice_list = allDevices[i].create_sub_devices<sycl::info::partition_property::partition_by_affinity_domain>(sycl::info::partition_affinity_domain::numa);
	_sycl_devices.insert(std::end(_sycl_devices),
			     std::begin(subdevice_list), std::end(subdevice_list));
	for (auto sub_device : subdevice_list) {
	  add_properties( sub_device );
	  _num_devices++;
	}
      } else {
	_sycl_devices.push_back( allDevices[i] );
	add_properties( allDevices[i] );
	_num_devices++;
      }
    } // for i
#endif // GERYON_NUMA_FISSION

    return UCL_SUCCESS;
  }

  int UCL_Device::create_context() {
    _context = new sycl::context(*_sycl_device, sycl_asynchandler);
    if (_context == nullptr) {
      #ifndef UCL_NO_EXIT
      std::cerr << "UCL Error: Could not access accelerator number " << _device
                << " for use.\n";
      UCL_GERYON_EXIT;
      #endif
      return UCL_ERROR;
    }
    push_command_queue();
    _default_cq=0;
    return UCL_SUCCESS;
  }

  void UCL_Device::add_properties(sycl::device& device_list) {
    SYCLProperties op;

    op.name            = device_list.get_info<sycl::info::device::name>();
    op.global_mem      = device_list.get_info<sycl::info::device::global_mem_size>();
    op.shared_mem      = device_list.get_info<sycl::info::device::local_mem_size>();
    op.device_type     = device_list.get_info<sycl::info::device::device_type>();
    op.compute_units   = device_list.get_info<sycl::info::device::max_compute_units>();
    op.clock           = device_list.get_info<sycl::info::device::max_clock_frequency>();
    op.work_group_size = device_list.get_info<sycl::info::device::max_work_group_size>();

    auto item_size = device_list.get_info<sycl::info::device::max_work_item_sizes>();
    op.work_item_size[0]  = item_size[0];
    op.work_item_size[1]  = item_size[1];
    op.work_item_size[2]  = item_size[2];    

    op.alignment       = device_list.get_info<sycl::info::device::mem_base_addr_align>();
    op.alignment      /= 8;

    op.preferred_vector_width32 = device_list.get_info<sycl::info::device::preferred_vector_width_float>();
    op.preferred_vector_width64 = device_list.get_info<sycl::info::device::preferred_vector_width_double>();
    op.has_double_precision = device_list.has(sycl::aspect::fp64);
    op.timer_resolution = device_list.get_info<sycl::info::device::profiling_timer_resolution>();

    op.sycl_device_version = device_list.get_info<sycl::info::device::version>();
    op.max_sub_devices = device_list.get_info<sycl::info::device::partition_max_sub_devices>();

    op.has_subgroup_support=false;
    std::vector<size_t> subg_sizes = device_list.get_info<sycl::info::device::sub_group_sizes>();
    if (subg_sizes.size() > 0) op.has_subgroup_support = true;

    _properties.push_back(op);
  }

  std::string UCL_Device::platform_name() {
    std::string ans = _sycl_platform.get_info<sycl::info::platform::vendor>();
    ans += ' ';
    ans += _sycl_platform.get_info<sycl::info::platform::name>();
    ans += ' ';
    ans += _sycl_platform.get_info<sycl::info::platform::version>();

    return ans;
  }

  // Set the SYCL device to the specified device number
  int UCL_Device::set(int num) {
    _device=num;
    _sycl_device=&(_sycl_devices[_device]);
    return create_context();
  }

  // List all devices from all platforms along with all properties
  void UCL_Device::print_all(std::ostream &out) {
    // --- loop through the platforms
    for (int n=0; n<_num_platforms; n++) {

      set_platform(n);

      out << "\nPlatform " << n << ":\n";

      if (num_devices() == 0)
        out << "There is no device supporting SYCL\n";
      for (int i=0; i<num_devices(); ++i) {
        out << "\nDevice " << i << ": \"" << name(i).c_str() << "\"\n";
        out << "  Type of device:                                "
            << device_type_name(i).c_str() << std::endl;
	out << "  Supported SYCL Version:                        "
          << _properties[i].sycl_device_version << std::endl;
        out << "  Double precision support:                      ";
        if (double_precision(i))
          out << "Yes\n";
        else
          out << "No\n";
        out << "  Total amount of global memory:                 "
            << gigabytes(i) << " GB\n";
        out << "  Number of compute units/multiprocessors:       "
            << _properties[i].compute_units << std::endl;
        //out << "  Number of cores:                               "
        //    << cores(i) << std::endl;
        out << "  Total amount of local/shared memory per block: "
            << _properties[i].shared_mem << " bytes\n";
        //out << "  Total number of registers available per block: "
        //    << _properties[i].regsPerBlock << std::endl;
        //out << "  Warp size:                                     "
        //    << _properties[i].warpSize << std::endl;
        out << "  Maximum group size (# of threads per block)    "
            << _properties[i].work_group_size << std::endl;
        out << "  Maximum item sizes (# threads for each dim)    "
            << _properties[i].work_item_size[0] << " x "
            << _properties[i].work_item_size[1] << " x "
            << _properties[i].work_item_size[2] << std::endl;
        //out << "  Maximum sizes of each dimension of a grid:     "
        //    << _properties[i].maxGridSize[0] << " x "
        //    << _properties[i].maxGridSize[1] << " x "
        //    << _properties[i].maxGridSize[2] << std::endl;
        //out << "  Maximum memory pitch:                          "
        //    << _properties[i].memPitch) << " bytes\n";
        //out << "  Texture alignment:                             "
        //    << _properties[i].textureAlignment << " bytes\n";
        out << "  Clock rate:                                    "
            << clock_rate(i) << " GHz\n";
        //out << "  Concurrent copy and execution:                 ";
        out << "  Maximum subdevices from fission:               "
            << max_sub_devices(i) << std::endl;
        out << "  Shared memory system:                          ";
        if (shared_memory(i))
          out << "Yes\n";
        else
          out << "No\n";
        out << "  Subgroup support:                              ";
        if (_properties[i].has_subgroup_support)
          out << "Yes\n";
        else
          out << "No\n";
      }
    }
  }

  int UCL_Device::auto_set_platform(const enum UCL_DEVICE_TYPE type,
                                    const std::string vendor,
                                    const int ndevices,
                                    const int first_device) {
    if (_num_platforms < 2) return set_platform(0);

    int last_device = -1;
    if (first_device > -1) {
      if (ndevices)
        last_device = first_device + ndevices - 1;
      else
        last_device = first_device;
    }

    bool vendor_match=false;
    bool type_match=false;
    unsigned int max_cus=0;
    int best_platform=0;

    std::string vendor_upper=vendor;
    for (int i=0; i<vendor.length(); i++)
      if (vendor_upper[i]<='z' && vendor_upper[i]>='a')
        vendor_upper[i]=toupper(vendor_upper[i]);

    for (int n=0; n<_num_platforms; n++) {
      set_platform(n);
      if (last_device > -1 && last_device >= num_devices()) continue;
      if (ndevices > num_devices()) continue;

      int first_id=0;
      int last_id=num_devices()-1;
      if (last_device > -1) {
        first_id=first_device;
        last_id=last_device;
      }

      if (vendor_upper!="") {
        std::string pname = platform_name();
        for (int i=0; i<pname.length(); i++)
          if (pname[i]<='z' && pname[i]>='a')
            pname[i]=toupper(pname[i]);

        if (pname.find(vendor_upper)!=std::string::npos) {
          if (vendor_match == false) {
            best_platform=n;
            max_cus=0;
            vendor_match=true;
          }
        } else if (vendor_match)
          continue;
      }

      if (type != UCL_DEFAULT) {
        bool ptype_matched=false;
        for (int d=first_id; d<=last_id; d++) {
          if (type==device_type(d)) {
            if (type_match == false) {
              best_platform=n;
              max_cus=0;
              type_match=true;
              ptype_matched=true;
            }
          }
        }
        if (type_match==true && ptype_matched==false)
          continue;
      }

      for (int d=first_id; d<=last_id; d++) {
        if (cus(d) > max_cus) {
          best_platform=n;
          max_cus=cus(d);
        }
      }
    }
    return set_platform(best_platform);
  }

} // namespace ucl_sycl

#endif
