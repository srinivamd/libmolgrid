cmake_minimum_required (VERSION 3.18)
if (DEFINED USE_ROCM)
  project(LibMolGrid LANGUAGES C CXX)
else()
  project(LibMolGrid LANGUAGES C CXX CUDA)
endif()

#version number
set (VERSION_MAJOR 0)
set (VERSION_MINOR 1)
set (VERSION_PATCH 0)

set (CMAKE_VERBOSE_MAKEFILE ON CACHE BOOL "ON")

#set default build type
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(BUILD_SHARED "Build shared library" ON)
option(BUILD_STATIC "Build static library" ON)

if(${BUILD_CONTAINED_PYTHON}) 
  #if make a mostly statically linked python module, use static boost
   set(BUILD_STATIC ON)
   set(Boost_USE_STATIC_LIBS ON)
endif()

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/Modules;${PROJECT_SOURCE_DIR}/cmake")
# get git hash
include(cmake/git_revision.cmake)

# guard against in-source builds and bad build-type strings
include(cmake/safeguards.cmake)

# place binaries and libraries according to GNU standards
include(GNUInstallDirs)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})

#dependencies
if (DEFINED USE_ROCM)
	find_package( ROCM REQUIRED CONFIG QUIET PATHS ${USE_ROCM} )
  if( NOT ROCM_FOUND )
    message(STATUS "ROCm not installed... done")
  else()
    message(STATUS "Found ROCm installed... done")
  endif()
else()
  find_package(CUDA REQUIRED)
endif()
find_package(Boost REQUIRED COMPONENTS regex unit_test_framework program_options system filesystem iostreams)
find_package(OpenBabel3 REQUIRED)
include_directories(SYSTEM ${OPENBABEL3_INCLUDE_DIR})
find_package(ZLIB)

# configure a header file to pass some of the CMake settings
# to the source code
configure_file (
  "${PROJECT_SOURCE_DIR}/cmake/config.h.in"
  "${PROJECT_BINARY_DIR}/include/libmolgrid/config.h"
  )

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

if (NOT DEFINED USE_ROCM)
if(CMAKE_CXX_COMPILER_ID MATCHES GNU)
    set(CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS} -Wall -Wno-unknown-pragmas -Werror")
    set(CMAKE_CXX_FLAGS_DEBUG   "-O0 -g3")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -g")
    set(CMAKE_CXX_FLAGS_PROFILE " -fprofile-arcs -ftest-coverage")
endif()

if(CUDA_VERSION_MAJOR LESS 11)
#compile for all major architectures >= 35
 set(CMAKE_CUDA_ARCHITECTURES 35 50 60 70 75)
else()
 set(CMAKE_CUDA_ARCHITECTURES 60 70 75)
endif()

set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -g -lineinfo")
set(CMAKE_CUDA_FLAGS_DEBUG "-O0 -g -G")
else()
	#  set(CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS} -v -Wall -Wno-unknown-pragmas  -Wno-unused-command-line-argument -D__CUDA_ARCH__ -D__HIP_PLATFORM_HCC__")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-unknown-pragmas  -Wno-unused-command-line-argument -D__HIP_DEVICE_COMPILE__ -D__HIP_PLATFORM_HCC__ -D__HIP_ROCclr__ -std=c++17 ")
  message(STATUS "DEBUG CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
endif()

# add the binary tree to the search path for include files
# so that we will find libmolgrid_config.h
include_directories("${PROJECT_BINARY_DIR}")
include_directories(BEFORE ${CMAKE_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
if (DEFINED USE_ROCM)
  include_directories(SYSTEM ${USE_ROCM}/include)
else()
  include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
endif()
include_directories(SYSTEM ${Boost_INCLUDE_DIR})

add_subdirectory(src)
add_subdirectory(python)
add_subdirectory(docs)

if (DEFINED USE_ROCM)
	#  target_compile_options(libmolgrid_shared PRIVATE  -D__HIP_ROCclr__ -std=c++11 -isystem /opt/rocm-4.1.0/llvm/lib/clang/12.0.0/include/.. -isystem /opt/rocm-4.1.0/hsa/include -D__HIP_ROCclr__ -isystem /opt/rocm-4.1.0/hip/include -O3)
include( ROCMSetupVersion )
include( ROCMCreatePackage )
include( ROCMInstallTargets )
include( ROCMPackageConfigHelpers )
include( ROCMInstallSymlinks )
endif()
  message(STATUS "DEBUG-2 CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")

# Append our library helper cmake path and the cmake path for hip (for convenience)
# Users may override HIP path by specifying their own in CMAKE_MODULE_PATH
list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake )

# NOTE:  workaround until llvm & hip cmake modules fixes symlink logic in their config files; remove when fixed
list( APPEND CMAKE_PREFIX_PATH ${USE_ROCM}/llvm ${USE_ROCM}/hip )

if(NOT SKIP_LIBRARY)
  message(STATUS "DEBUG-3 CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
  set( Tensile_ARCHITECTURE "all" CACHE STRING "Tensile to use which architecture?")
  set_property( CACHE Tensile_ARCHITECTURE PROPERTY STRINGS all gfx803 gfx900 gfx906:xnack- gfx908:xnack-)
  # -DTensile_ARCHITECTURE= set by install.sh for builds with and without tensile

  # Detect if target ID syntax if supported for default all AMDGPU_TARGETS list
  if(CMAKE_CXX_COMPILER MATCHES ".*/hipcc$" )
    execute_process(COMMAND ${CMAKE_CXX_COMPILER} "--help"
      OUTPUT_VARIABLE CXX_OUTPUT
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_STRIP_TRAILING_WHITESPACE)
    string(REGEX MATCH ".mcode\-object\-version" TARGET_ID_SUPPORT ${CXX_OUTPUT})
  endif()
  if(TARGET_ID_SUPPORT)
    set(target_list "gfx803;gfx900;gfx906:xnack-;gfx908:xnack-")
  else()
    set(target_list "gfx803;gfx900;gfx906;gfx908")
  endif()
  set(ALL_TARGETS ${target_list} CACHE STRING "List of specific machine types for library to target")

  # gpu architectures to build for
  if (Tensile_ARCHITECTURE)
    if( Tensile_ARCHITECTURE STREQUAL "all" )
      set( gpus ${ALL_TARGETS})
    else()
      set( gpus ${Tensile_ARCHITECTURE})
    endif()
    # match tensile if architecture is set
    set( AMD_GPUS ${gpus} )
    # must FORCE set this AMDGPU_TARGETS before any find_package( hip ...), via tensile or in this file
    # to override CACHE var and set cuda-gpu-arch flags via hip-config.cmake hip::device dependency
    set( AMDGPU_TARGETS "${AMD_GPUS}" CACHE STRING "AMD GPU targets to compile for" FORCE )
  endif()
  message(STATUS "DEBUG-4 CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")

endif()

# Find HIP dependencies
if( CMAKE_CXX_COMPILER MATCHES ".*/hipcc$" )
	find_package( hip REQUIRED CONFIG PATHS ${USE_ROCM} )
endif( )

message(STATUS "Using AMDGPU_TARGETS: ${AMDGPU_TARGETS}")


# enable testing
include(CTest)
enable_testing()
# define tests
add_subdirectory(test)
