########################################################################
# SYCL requires C++17
if(CMAKE_CXX_STANDARD LESS 17)
  message(FATAL_ERROR "The SYCL package requires the C++ standard to be set to at least C++17")
endif()

########################################################################
# Downlaod and install Kokkos/mdspan
include(FetchContent)
FetchContent_Declare(mdspan
  GIT_REPOSITORY https://github.com/kokkos/mdspan.git
  GIT_TAG        stable
)
FetchContent_MakeAvailable( mdspan )
########################################################################
# consistency checks and Sycl options/settings required by LAMMPS
if(Sycl_ENABLE_CUDA)
  message(STATUS "SYCL: Enabling CUDA LAMBDA function support")
  set(Sycl_ENABLE_CUDA_LAMBDA ON CACHE BOOL "" FORCE)
endif()
# Adding OpenMP compiler flags without the checks done for
# BUILD_OMP can result in compile failures. Enforce consistency.
if(Sycl_ENABLE_OPENMP)
  if(NOT BUILD_OMP)
    message(FATAL_ERROR "Must enable BUILD_OMP with Sycl_ENABLE_OPENMP")
  else()
    # NVHPC does not seem to provide a detectable OpenMP version, but is far beyond version 3.1
    if((OpenMP_CXX_VERSION VERSION_LESS 3.1) AND NOT (CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC"))
      message(FATAL_ERROR "Compiler must support OpenMP 3.1 or later with Sycl_ENABLE_OPENMP")
    endif()
  endif()
endif()
########################################################################

if(BUILD_SHARED_LIBS_WAS_ON)
  set(BUILD_SHARED_LIBS ON)
endif()

target_compile_definitions(lammps PUBLIC $<BUILD_INTERFACE:LMP_SYCL>)

set(SYCL_PKG_SOURCES_DIR ${LAMMPS_SOURCE_DIR}/SYCL)
set(SYCL_PKG_SOURCES ${SYCL_PKG_SOURCES_DIR}/sycl.cpp
                     ${SYCL_PKG_SOURCES_DIR}/atom_sycl.cpp
                     ${SYCL_PKG_SOURCES_DIR}/atom_map_sycl.cpp
                     ${SYCL_PKG_SOURCES_DIR}/atom_vec_sycl.cpp
                     ${SYCL_PKG_SOURCES_DIR}/comm_sycl.cpp
                     ${SYCL_PKG_SOURCES_DIR}/comm_tiled_sycl.cpp
                     ${SYCL_PKG_SOURCES_DIR}/min_sycl.cpp
                     ${SYCL_PKG_SOURCES_DIR}/min_linesearch_sycl.cpp
                     ${SYCL_PKG_SOURCES_DIR}/neighbor_sycl.cpp
                     ${SYCL_PKG_SOURCES_DIR}/neigh_list_sycl.cpp
                     ${SYCL_PKG_SOURCES_DIR}/neigh_bond_sycl.cpp
                     ${SYCL_PKG_SOURCES_DIR}/fix_nh_sycl.cpp
                     ${SYCL_PKG_SOURCES_DIR}/nbin_sycl.cpp
                     ${SYCL_PKG_SOURCES_DIR}/npair_sycl.cpp
                     ${SYCL_PKG_SOURCES_DIR}/npair_halffull_sycl.cpp
                     ${SYCL_PKG_SOURCES_DIR}/domain_sycl.cpp
                     ${SYCL_PKG_SOURCES_DIR}/modify_sycl.cpp)

if(PKG_KSPACE)
  list(APPEND SYCL_PKG_SOURCES ${SYCL_PKG_SOURCES_DIR}/fft3d_sycl.cpp
                                 ${SYCL_PKG_SOURCES_DIR}/grid3d_sycl.cpp
                                 ${SYCL_PKG_SOURCES_DIR}/remap_sycl.cpp)
  if(Sycl_ENABLE_CUDA)
    if(NOT (FFT STREQUAL "KISS"))
      target_compile_definitions(lammps PRIVATE -DFFT_CUFFT)
      target_link_libraries(lammps PRIVATE cufft)
    endif()
  elseif(Sycl_ENABLE_HIP)
    if(NOT (FFT STREQUAL "KISS"))
      target_compile_definitions(lammps PRIVATE -DFFT_HIPFFT)
      target_link_libraries(lammps PRIVATE hipfft)
    endif()
  endif()
endif()

set_property(GLOBAL PROPERTY "SYCL_PKG_SOURCES" "${SYCL_PKG_SOURCES}")

# detects styles which have SYCL version
RegisterStylesExt(${SYCL_PKG_SOURCES_DIR} sycl SYCL_PKG_SOURCES)

# register sycl-only styles
RegisterNBinStyle(${SYCL_PKG_SOURCES_DIR}/nbin_sycl.h)
RegisterNPairStyle(${SYCL_PKG_SOURCES_DIR}/npair_sycl.h)
RegisterNPairStyle(${SYCL_PKG_SOURCES_DIR}/npair_halffull_sycl.h)

get_property(SYCL_PKG_SOURCES GLOBAL PROPERTY SYCL_PKG_SOURCES)

target_sources(lammps PRIVATE ${SYCL_PKG_SOURCES})
target_include_directories(lammps PUBLIC $<BUILD_INTERFACE:${SYCL_PKG_SOURCES_DIR}>)
