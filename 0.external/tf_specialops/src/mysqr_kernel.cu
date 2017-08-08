#include "cuda_helper.h"

namespace mysqr_gpu
{
  template <class T>
  __global__ void mysqr_kernel(T* out, const T* in, const int N)
  {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    if( x >= N )
      return;

    out[x] = in[x]*in[x];
  }

  template <class T>
  __global__ void mysqrgrad_kernel(T* out, const T* in, const T* grad, const int N)
  {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    if( x >= N )
      return;

    out[x] = 2*in[x]*grad[x];
  }


}
using namespace mysqr_gpu;

template <class T>
void mysqr_gpu_kernel(const cudaStream_t& stream, T* out, const T* in, const int N)
{
  dim3 block(128,1,1);
  dim3 grid;
  grid.x = divup(N,block.x);

  mysqr_kernel<T><<<grid,block,0,stream>>>(out, in, N);
}
template void mysqr_gpu_kernel<float>(const cudaStream_t&, float*, const float*, const int);
template void mysqr_gpu_kernel<double>(const cudaStream_t&, double*, const double*, const int);



template <class T>
void mysqrgrad_gpu_kernel(const cudaStream_t& stream, T* out, const T* in, const T* grad, const int N)
{
  dim3 block(128,1,1);
  dim3 grid;
  grid.x = divup(N,block.x);

  mysqrgrad_kernel<T><<<grid,block,0,stream>>>(out, in, grad, N);
}
template void mysqrgrad_gpu_kernel<float>(const cudaStream_t&, float*, const float*, const float*, const int);
template void mysqrgrad_gpu_kernel<double>(const cudaStream_t&, double*, const double*, const double*, const int);



