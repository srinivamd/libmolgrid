/*
 * common.h
 *
 * Utility functions and definitions
 */

#ifndef COMMON_H_
#define COMMON_H_

#include <memory>
#include <cstring>
#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>

#ifdef __HIPCC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#define CUDA_DEVICE_MEMBER __device__
#else
#define CUDA_CALLABLE_MEMBER
#define CUDA_DEVICE_MEMBER
#endif

//called in device code to perform a parallel operation
#define LMG_CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

// CUDA: use 512 threads per block
#define LMG_CUDA_NUM_THREADS 512
#define LMG_CUDA_BLOCKDIM 8
#define LOG2_WARP_SIZE 5U
#define WARP_SIZE (1U << LOG2_WARP_SIZE)

// CUDA: number of blocks for threads.
#define LMG_GET_BLOCKS(N) ((unsigned(N) + LMG_CUDA_NUM_THREADS - 1) / LMG_CUDA_NUM_THREADS)
// CUDA: combined with GET_BLOCKS, number of threads
#define LMG_GET_THREADS(N) min(N,LMG_CUDA_NUM_THREADS)

#ifndef __CUDA_ARCH__
#define LMG_CUDA_CHECK(condition) \
  /* Code block avoids redefinition of hipError_t error */ \
  do { \
    hipError_t error = condition; \
    if(error != hipSuccess) {                                          \
        std::cerr << __FILE__ << ":" << __LINE__ << ": " << hipGetErrorString(error); throw std::runtime_error(std::string("CUDA Error: ")+hipGetErrorString(error)); } \
  } while (0)
#else
// probably don't want to make API calls on the device.
#define LMG_CUDA_CHECK(condition) condition
#endif


#endif /* COMMON_H_ */
