#include "cuda_helper.h"

namespace median3x3downsample_internal
{
  template <class T>
  __global__ void median3x3downsample_kernel(
      T* out, const T* in,
      int z_size, 
      int out_x_size, int out_y_size, int out_xy_size,
      int in_x_size, int in_y_size, int in_xy_size )
  {
    int z = blockIdx.z*blockDim.z + threadIdx.z;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    if( x >= out_x_size || y >= out_y_size || z >= z_size )
      return;

    T value[9];
    int value_idx = 0;
    for( int dy = -1; dy <= 1; ++dy )
    for( int dx = -1; dx <= 1; ++dx )
    {
      int x_ = min(in_x_size-1,max(0,2*x+dx));
      int y_ = min(in_y_size-1,max(0,2*y+dy));
      value[value_idx++] = in[z*in_xy_size+y_*in_x_size+x_];
    }
    {
      for(int j = 1; j < 9; ++j)
      {
        if( value[0] > value[j] )
        {
          T tmp = value[0];
          value[0] = value[j];
          value[j] = tmp;
        }
      }
      for(int j = 2; j < 9; ++j)
      {
        if( value[1] > value[j] )
        {
          T tmp = value[1];
          value[1] = value[j];
          value[j] = tmp;
        }
      }
      for(int j = 3; j < 9; ++j)
      {
        if( value[2] > value[j] )
        {
          T tmp = value[2];
          value[2] = value[j];
          value[j] = tmp;
        }
      }
      for(int j = 4; j < 9; ++j)
      {
        if( value[3] > value[j] )
        {
          T tmp = value[3];
          value[3] = value[j];
          value[j] = tmp;
        }
      }
      for(int j = 5; j < 9; ++j)
      {
        if( value[4] > value[j] )
        {
          T tmp = value[4];
          value[4] = value[j];
          value[j] = tmp;
        }
      }
    }
    int out_idx = z*out_xy_size + y*out_x_size + x;
    out[out_idx] = value[4];
  }
}
using namespace median3x3downsample_internal;

template <class T>
void median3x3downsample_gpu( 
    const cudaStream_t& stream,
    T* out, const T* in, 
    int z_size, 
    int in_y_size, int in_x_size )
{
  int out_x_size = divup(in_x_size,2);
  int out_y_size = divup(in_y_size,2);
  int out_xy_size = out_x_size*out_y_size;
  int in_xy_size = in_x_size*in_y_size;
  dim3 block(32,4,1);
  dim3 grid;
  grid.x = divup(out_x_size,block.x);
  grid.y = divup(out_y_size,block.y);
  grid.z = divup(z_size,block.z);

  median3x3downsample_kernel<T><<<grid,block,0,stream>>>(
      out, in, 
      z_size,
      out_x_size, out_y_size, out_xy_size,
      in_x_size, in_y_size, in_xy_size
      );
}
template void median3x3downsample_gpu<float>(const cudaStream_t&, float*, const float*, int, int, int);
template void median3x3downsample_gpu<double>(const cudaStream_t&, double*, const double*, int, int, int);

