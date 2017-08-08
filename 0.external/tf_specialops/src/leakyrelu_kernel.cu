#include "cuda_helper.h"

namespace leakyrelu_kernel_internal
{

  __device__ __forceinline__ float Max(const float a, const float b)
  {
    return fmaxf(a,b);
  }
  __device__ __forceinline__ double Max(const double a, const double b)
  {
    return fmax(a,b);
  }

  template <class T>
  __global__ void leakyrelu_kernel(T* out, const T* in, const T leak, const int size )
  {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if( x >= size )
      return;

    const T tmp = in[x];
    out[x] = Max(leak*tmp,tmp);
  }

  template <class T>
  __global__ void leakyrelu_grad_kernel(T* out, const T* in, const T* grad, const T leak, const int size )
  {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if( x >= size )
      return;

    const T tmp = in[x];
    const T leak_tmp = leak*tmp;
    if( tmp >= leak_tmp )
      out[x] = grad[x];
    else
      out[x] = leak*grad[x];
  }


} 
using namespace leakyrelu_kernel_internal;



template <class T>
void leakyrelu_gpu( 
    const cudaStream_t& stream,
    T* out, const T* in, 
    const T leak,
    const int64_t size )
{

  dim3 block(256,1,1);
  dim3 grid;
  grid.x = divup(size,block.x);
  grid.y = 1;
  grid.z = 1;

  leakyrelu_kernel<T><<<grid,block,0,stream>>>(out, in, leak, size);
  CHECK_CUDA_ERROR;

}
template void leakyrelu_gpu<float>(const cudaStream_t&, float*, const float*, const float, int64_t);
template void leakyrelu_gpu<double>(const cudaStream_t&, double*, const double*, const double, int64_t);


template <class T>
void leakyrelu_grad_gpu( 
    const cudaStream_t& stream,
    T* out, const T* in, const T* grad, const T leak,
    const int64_t size )
{

  dim3 block(256,1,1);
  dim3 grid;
  grid.x = divup(size,block.x);
  grid.y = 1;
  grid.z = 1;

  leakyrelu_grad_kernel<T><<<grid,block,0,stream>>>(out, in, grad, leak, size);
  CHECK_CUDA_ERROR;

}
template void leakyrelu_grad_gpu<float>(const cudaStream_t&, float*, const float*, const float*, const float, int64_t);
template void leakyrelu_grad_gpu<double>(const cudaStream_t&, double*, const double*, const double*, const double, int64_t);


