#include "cuda_helper.h"

namespace replacenonfinite_kernel_internal
{
  template <class T>
  __global__ void replacenonfinite_kernel(T* out, const T* in, const T value, const int size )
  {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if( x >= size )
      return;

    T tmp = in[x];
    if( isfinite(tmp) )
      out[x] = tmp;
    else
      out[x] = value;
  }

  template <class T>
  __global__ void replacenonfinite_grad_kernel(T* out, const T* in, const T* grad, const int size )
  {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if( x >= size )
      return;

    T tmp = in[x];
    if( isfinite(tmp) )
      out[x] = grad[x];
    else
      out[x] = T(0);
  }


} 
using namespace replacenonfinite_kernel_internal;



template <class T>
void replacenonfinite_gpu( 
    const cudaStream_t& stream,
    T* out, const T* in, 
    const T value,
    const int64_t size )
{

  dim3 block(256,1,1);
  dim3 grid;
  grid.x = divup(size,block.x);
  grid.y = 1;
  grid.z = 1;

  replacenonfinite_kernel<T><<<grid,block,0,stream>>>(out, in, value, size);
  CHECK_CUDA_ERROR;

}
template void replacenonfinite_gpu<float>(const cudaStream_t&, float*, const float*, const float, int64_t);
template void replacenonfinite_gpu<double>(const cudaStream_t&, double*, const double*, const double, int64_t);


template <class T>
void replacenonfinite_grad_gpu( 
    const cudaStream_t& stream,
    T* out, const T* in, const T* grad,
    const int64_t size )
{

  dim3 block(256,1,1);
  dim3 grid;
  grid.x = divup(size,block.x);
  grid.y = 1;
  grid.z = 1;

  replacenonfinite_grad_kernel<T><<<grid,block,0,stream>>>(out, in, grad, size);
  CHECK_CUDA_ERROR;

}
template void replacenonfinite_grad_gpu<float>(const cudaStream_t&, float*, const float*, const float*, int64_t);
template void replacenonfinite_grad_gpu<double>(const cudaStream_t&, double*, const double*, const double*, int64_t);


