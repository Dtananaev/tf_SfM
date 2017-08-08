#define EIGEN_USE_GPU
#include "cuda_helper.h"
#include "Eigen/Geometry"

namespace depthtonormals_kernel_internal
{

  template <class T>
  __device__ inline void compute3dPoint( 
      Eigen::Matrix<T,3,1>& p_3d, 
      const int& x, const int& y, 
      const T& depth, 
      const T& inv_fx, const T& inv_fy,
      const T& cx, const T& cy )
  {
    p_3d << (x+T(0.5)-cx)*inv_fx*depth, 
            (y+T(0.5)-cy)*inv_fy*depth, 
            depth;
  }



  template <class T, bool INVERSE_DEPTH>
  __global__ void depthtonormals_kernel(
      T* out, const T* depth, const T* intrinsics,
      int x_size, int y_size, int z_size )
  {
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if( x >= x_size || y >= y_size || z >= z_size )
      return;

    const T inv_fx = 1/(intrinsics[4*z+0]*x_size);
    const T inv_fy = 1/(intrinsics[4*z+1]*y_size);
    const T cx = 1/(intrinsics[4*z+2]*x_size);
    const T cy = 1/(intrinsics[4*z+3]*y_size);

    typedef Eigen::Matrix<T,3,1> Vec3;
    const int xy_size = x_size*y_size;
    const T* depthmap = depth+z*xy_size;
    T* normal = out+3*z*xy_size;
#define DEPTH(y,x) depthmap[(y)*x_size+(x)]
#define NORMAL(c,y,x) normal[(c)*xy_size+(y)*x_size+(x)]
    if( x == 0 || y == 0 || x == x_size-1 || y == y_size-1)
    {
      NORMAL(0,y,x) = NAN;
      NORMAL(1,y,x) = NAN;
      NORMAL(2,y,x) = NAN;
    }
    else
    {
      T d = DEPTH(y,x);
      T d_y0 = DEPTH(y-1,x);
      T d_x0 = DEPTH(y,x-1);
      T d_y1 = DEPTH(y+1,x);
      T d_x1 = DEPTH(y,x+1);
      if( INVERSE_DEPTH )
      {
        d = 1/d;
        d_y0 = 1/d_y0;
        d_x0 = 1/d_x0;
        d_y1 = 1/d_y1;
        d_x1 = 1/d_x1;
      }
      
      if( d <= 0 || !isfinite(d) || 
          d_y0 <= 0 || !isfinite(d_y0) || 
          d_x0 <= 0 || !isfinite(d_x0) || 
          d_y1 <= 0 || !isfinite(d_y1) || 
          d_x1 <= 0 || !isfinite(d_x1))
      {
        NORMAL(0,y,x) = NAN;
        NORMAL(1,y,x) = NAN;
        NORMAL(2,y,x) = NAN;
      }
      else
      {
        Vec3 p, p_y0, p_x0, p_y1, p_x1;
        compute3dPoint(p, x, y, d, inv_fx, inv_fy, cx, cy);
        compute3dPoint(p_y0, x, y-1, d_y0, inv_fx, inv_fy, cx, cy);
        compute3dPoint(p_x0, x-1, y, d_x0, inv_fx, inv_fy, cx, cy);
        compute3dPoint(p_y1, x, y+1, d_y1, inv_fx, inv_fy, cx, cy);
        compute3dPoint(p_x1, x+1, y, d_x1, inv_fx, inv_fy, cx, cy);

        Vec3 normals_vec1 = (p - p_x1).cross(p_y1 - p);
        Vec3 normals_vec0 = (p - p_x0).cross(p_y0 - p);
        normals_vec1.normalize();
        normals_vec0.normalize();
        
        Vec3 normals_vec = (normals_vec1 + normals_vec0);
        normals_vec.normalize();
        
        NORMAL(0,y,x) = normals_vec.x();
        NORMAL(1,y,x) = normals_vec.y();
        NORMAL(2,y,x) = normals_vec.z();
      }
    }

#undef DEPTH
#undef NORMAL
  }

} 
using namespace depthtonormals_kernel_internal;



template <class T>
void depthtonormals_gpu( 
    const cudaStream_t& stream,
    T* out, const T* depth, const T* intrinsics,
    bool inverse_depth,
    int x_size, int y_size, int z_size )
{

  dim3 block(32,4,1);
  dim3 grid;
  grid.x = divup(x_size,block.x);
  grid.y = divup(y_size,block.y);
  grid.z = divup(z_size,block.z);

  if( inverse_depth )
    depthtonormals_kernel<T,true><<<grid,block,0,stream>>>(
        out, depth, intrinsics, x_size, y_size, z_size );
  else
    depthtonormals_kernel<T,false><<<grid,block,0,stream>>>(
        out, depth, intrinsics, x_size, y_size, z_size );
  

  CHECK_CUDA_ERROR;
}
template void depthtonormals_gpu<float>( 
    const cudaStream_t& stream,
    float* out, const float* depth, const float* intrinsics,
    bool inverse_depth,
    int x_size, int y_size, int z_size );
template void depthtonormals_gpu<double>(
    const cudaStream_t& stream,
    double* out, const double* depth, const double* intrinsics,
    bool inverse_depth,
    int x_size, int y_size, int z_size );

