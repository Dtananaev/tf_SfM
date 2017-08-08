#include "cuda_helper.h"

namespace scaleinvariantgrad_kernel_internal
{

  template <class T>
  __device__ __forceinline__ bool is_valid(const T& value)
  {
    return isfinite(value);
  }

  template <class T>
  __device__ __forceinline__ T dcenter(const T& center_value, const T& neighbour_value, const T& eps)
  {
    T sum_abs = std::abs(center_value) + std::abs(neighbour_value) + eps;
    T sign = -1;
    if( center_value < 0 )
      sign = 1;
    return -1/sum_abs + sign*(neighbour_value - center_value)/(sum_abs*sum_abs);
  }

  template <class T>
  __device__ __forceinline__ T dneighbour(const T& center_value, const T& neighbour_value, const T& eps)
  {
    T sum_abs = std::abs(center_value) + std::abs(neighbour_value) + eps;
    T sign = -1;
    if( neighbour_value < 0 )
      sign = 1;
    return 1/sum_abs + sign*(neighbour_value - center_value)/(sum_abs*sum_abs);
  }


  template <class T>
  __global__ void computeForward( 
      T* out, const T* input,
      int x_size, int y_size, int z_size, T eps,
      const int* deltas, const T* weights, int max_comparisons )
  {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if( x >= x_size || y >= y_size || z >= z_size )
      return;

    const int xy_size = x_size*y_size;

#define INPUT(x,y,z) input[(z)*xy_size+(y)*x_size+(x)]
#define OUT(x,y,z) out[(z)*xy_size+(y)*x_size+(x)]

    const T value0 = INPUT(x,y,z);
    T grad_x = 0;
    T grad_y = 0;

    for( int comparison = 0; comparison < max_comparisons; ++comparison )
    {
      int delta = deltas[comparison];
      T weight = weights[comparison];

      T valuex, valuey;
      if( x+delta >= 0 && x+delta < x_size )
        valuex = INPUT(x+delta,y,z);
      else
        valuex = value0;

      if( y+delta >= 0 && y+delta < y_size )
        valuey = INPUT(x,y+delta,z);
      else
        valuey = value0;

      grad_x += weight*(valuex-value0)/(std::abs(value0)+std::abs(valuex)+eps);
      grad_y += weight*(valuey-value0)/(std::abs(value0)+std::abs(valuey)+eps);
    }

    OUT(x,y,2*z+0) = grad_x;
    OUT(x,y,2*z+1) = grad_y;

#undef INPUT
#undef OUT
  }




  template <class T>
  __global__ void computeBackward( 
      T* out, const T* input_data, const T* grad, 
      int x_size, int y_size, int z_size, T eps,
      const int* deltas, const T* weights, int max_comparisons )
  {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if( x >= x_size || y >= y_size || z >= z_size )
      return;

    const int xy_size = x_size*y_size;

#define INPUT(x,y,z) input_data[(z)*xy_size+(y)*x_size+(x)]
#define OUT(x,y,z) out[(z)*xy_size+(y)*x_size+(x)]
#define GRAD(x,y,z) grad[(z)*xy_size+(y)*x_size+(x)]

    T value0_diff = 0;
    const T value0 = INPUT(x,y,z);

    if( is_valid(value0) )
    {
      for( int comparison = 0; comparison < max_comparisons; ++comparison )
      {
        int delta = deltas[comparison];
        T weight = weights[comparison];

        T value0_diff_tmp = 0;

          // compute the backpropagated gradient from the x component
          if( x+delta >= 0 && x+delta < x_size )
          {
            const T valuex = INPUT(x+delta,y,z);
            if( is_valid(valuex) )
            {
              T tmp = dcenter(value0, valuex, eps);
              value0_diff_tmp += tmp * GRAD(x,y,2*z+0);
            }
          }
          if( x-delta >= 0 && x-delta < x_size )
          {
            const T valuex = INPUT(x-delta,y,z);
            if( is_valid(valuex) )
            {
              T tmp = dneighbour(valuex, value0, eps);
              value0_diff_tmp += tmp * GRAD(x-delta,y,2*z+0);
            }
          } 

          // compute the backpropagated gradient from the y component
          if( y+delta >= 0 && y+delta < y_size )
          {
            const T valuey = INPUT(x,y+delta,z);
            if( is_valid(valuey) )
            {
              T tmp = dcenter(value0, valuey, eps);
              value0_diff_tmp += tmp * GRAD(x,y,2*z+1);
            }
          }
          if( y-delta >= 0 && y-delta < y_size )
          {
            const T valuey = INPUT(x,y-delta,z);
            if( is_valid(valuey) )
            {
              T tmp = dneighbour(valuey, value0, eps);
              value0_diff_tmp += tmp * GRAD(x,y-delta,2*z+1);
            }
          } 
          value0_diff += weight*value0_diff_tmp;
       }
    }

    if( !isfinite(value0_diff) )
      value0_diff = 0;
    OUT(x,y,z) = value0_diff;

#undef INPUT
#undef OUT
#undef GRAD

  }

} 
using namespace scaleinvariantgrad_kernel_internal;



template <class T>
void scaleinvariantgrad_gpu( 
    const cudaStream_t& stream,
    T* out, const T* in, 
    const int* deltas,
    const T* weights,
    int num_deltas,
    T epsilon,
    int x_size, int y_size, int z_size )
{

  dim3 block(32,4,1);
  dim3 grid;
  grid.x = divup(x_size,block.x);
  grid.y = divup(y_size,block.y);
  grid.z = divup(z_size,block.z);

  computeForward<T><<<grid,block,0,stream>>>(
      out, in,
      x_size, y_size, z_size, 
      epsilon,
      deltas, weights, num_deltas );
  CHECK_CUDA_ERROR

}
template void scaleinvariantgrad_gpu<float>(const cudaStream_t&, float*, const float*, const int*, const float*, int, float, int, int, int);
template void scaleinvariantgrad_gpu<double>(const cudaStream_t&, double*, const double*, const int*, const double*, int, double, int, int, int);



template <class T>
void scaleinvariantgrad_grad_gpu( 
    const cudaStream_t& stream,
    T* out, const T* in, const T* grad, 
    const int* deltas, const T* weights, int num_deltas, T eps,
    int x_size, int y_size, int z_size )
{

  dim3 block(32,4,1);
  dim3 grid;
  grid.x = divup(x_size,block.x);
  grid.y = divup(y_size,block.y);
  grid.z = divup(z_size,block.z);

  computeBackward<T><<<grid,block,0,stream>>>( 
      out, in, grad, 
      x_size, y_size, z_size, eps,
      deltas, weights, num_deltas );
  CHECK_CUDA_ERROR

}
template void scaleinvariantgrad_grad_gpu<float>(const cudaStream_t&, float*, const float*, const float*, const int*, const float*, int, float, int, int, int);
template void scaleinvariantgrad_grad_gpu<double>(const cudaStream_t&, double*, const double*, const double*, const int*, const double*, int, double, int, int, int);

