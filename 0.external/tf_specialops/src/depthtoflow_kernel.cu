#define EIGEN_USE_GPU
#include "cuda_helper.h"
#include "rotation_format.h"
#include "Eigen/Core"
#include <cuda_runtime.h>

namespace depthtoflow_internal
{

  template <class T, class VEC2T, class VEC3T, class MAT3T>
  __device__ inline void compute_flow( 
      Eigen::MatrixBase<VEC2T>& flow,        // the flow vector
      const Eigen::MatrixBase<VEC2T>& p1,    // pixel coordinates in the first image with pixel centers at x.5, y.5
      const T depth,                         // depth of the point in the first image
      const Eigen::MatrixBase<VEC2T>& f,     // focal lengths
      const Eigen::MatrixBase<VEC2T>& inv_f, // reciprocal of focal lengths (1/f.x, 1/f.y)
      const Eigen::MatrixBase<VEC2T>& c,     // principal point coordinates, not pixel coordinates! pixel centers are shifted by 0.5
      const Eigen::MatrixBase<MAT3T>& R,     // rotation
      const Eigen::MatrixBase<VEC3T>& t      // translation
      ) 
  {
    EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(VEC2T, 2) 
    EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(VEC3T, 3) 
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(MAT3T, 3, 3) 
    typedef Eigen::Matrix<T,2,1> Vec2;
    typedef Eigen::Matrix<T,3,1> Vec3;
    // compute the 3d point in the coordinate frame of the first camera
    Vec2 tmp2 = (p1-c).cwiseProduct(inv_f);

    // transform the point to the coordinate frame of the second camera
    Vec3 p2 = R*(depth*tmp2.homogeneous()) + t;
    
    // project point to the image plane
    p2.x() = f.x()*(p2.x()/p2.z()) + c.x();
    p2.y() = f.y()*(p2.y()/p2.z()) + c.y();
    flow = p2.template topRows<2>() - p1;
  }

  template <class T, bool NORMALIZE_FLOW, bool INVERSE_DEPTH>
  __global__ void depthtoflow_kernel(
      T* out, const T* depth,
      const T* intrinsics,
      const T* rotation,
      const T* translation,
      int depth_x_size, int depth_y_size, int depth_z_size, int depth_xy_size,
      T inv_depth_x_size, T inv_depth_y_size )
  {
    typedef Eigen::Matrix<T,2,1> Vec2;
    typedef Eigen::Matrix<T,3,1> Vec3;
    typedef Eigen::Matrix<T,3,3> Mat3;
    int z = blockIdx.z*blockDim.z + threadIdx.z;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    if( x >= depth_x_size || y >= depth_y_size || z >= depth_z_size )
      return;

    Vec2 f, c;
    if( NORMALIZE_FLOW )
    {
      f.x() = intrinsics[4*z+0];
      f.y() = intrinsics[4*z+1];
      c.x() = intrinsics[4*z+2];
      c.y() = intrinsics[4*z+3];
    }
    else
    {
      f.x() = intrinsics[4*z+0]*depth_x_size;
      f.y() = intrinsics[4*z+1]*depth_y_size;
      c.x() = intrinsics[4*z+2]*depth_x_size;
      c.y() = intrinsics[4*z+3]*depth_y_size;
    }
    Vec2 inv_f(1/f.x(), 1/f.y());

    Eigen::Map<const Vec3> t(translation+3*z);
    Eigen::Map<const Mat3> R(rotation+9*z);

    const T* depthmap = depth+z*depth_xy_size;
    T* flow = out+2*z*depth_xy_size;
#define DEPTH(x,y) depthmap[(y)*depth_x_size+(x)]
#define FLOW(c,x,y) flow[(c)*depth_xy_size+(y)*depth_x_size+(x)]
    {
      Vec2 flow_vec(NAN,NAN);

      T d = DEPTH(x,y);
      if( INVERSE_DEPTH )
        d = 1/d;
      if( d > 0 && isfinite(d) )
      {
        Vec2 p1(x+T(0.5),y+T(0.5));
        if( NORMALIZE_FLOW )
        {
          p1.x() *= inv_depth_x_size;
          p1.y() *= inv_depth_y_size;
        }
        compute_flow(flow_vec, p1, d, f, inv_f, c, R, t);
      }

      FLOW(0,x,y) = flow_vec.x();
      FLOW(1,x,y) = flow_vec.y();
    }
#undef DEPTH
#undef FLOW
  }


}
using namespace depthtoflow_internal;



template <class T>
void depthtoflow_gpu( 
      const cudaStream_t& stream,
      T* out, 
      const T* depth, 
      const T* intrinsics,
      const T* rotation,
      const T* translation,
      int depth_x_size, int depth_y_size, int depth_z_size,
      bool normalize_flow,
      bool inverse_depth )
{
  dim3 block(32,4,1);
  dim3 grid;
  grid.x = divup(depth_x_size,block.x);
  grid.y = divup(depth_y_size,block.y);
  grid.z = divup(depth_z_size,block.z);

  if( normalize_flow )
  {
    if( inverse_depth )
    {
      depthtoflow_kernel<T,true,true><<<grid,block,0,stream>>>(
          out, depth,
          intrinsics,
          rotation,
          translation,
          depth_x_size, depth_y_size, depth_z_size, depth_x_size*depth_y_size,
          1.0/depth_x_size, 1.0/depth_y_size );
      CHECK_CUDA_ERROR
    }
    else
    {
      depthtoflow_kernel<T,true,false><<<grid,block,0,stream>>>(
          out, depth,
          intrinsics,
          rotation,
          translation,
          depth_x_size, depth_y_size, depth_z_size, depth_x_size*depth_y_size,
          1.0/depth_x_size, 1.0/depth_y_size );
      CHECK_CUDA_ERROR
    }
  }
  else
  {
    if( inverse_depth )
    {
      depthtoflow_kernel<T,false,true><<<grid,block,0,stream>>>(
          out, depth,
          intrinsics,
          rotation,
          translation,
          depth_x_size, depth_y_size, depth_z_size, depth_x_size*depth_y_size,
          1.0/depth_x_size, 1.0/depth_y_size );
      CHECK_CUDA_ERROR
    }
    else
    {
      depthtoflow_kernel<T,false,false><<<grid,block,0,stream>>>(
          out, depth,
          intrinsics,
          rotation,
          translation,
          depth_x_size, depth_y_size, depth_z_size, depth_x_size*depth_y_size,
          1.0/depth_x_size, 1.0/depth_y_size );
      CHECK_CUDA_ERROR
    }
  }
}
template void depthtoflow_gpu<float>(const cudaStream_t&, float*, const float*, const float*, const float*, const float*, int, int, int, bool, bool);
template void depthtoflow_gpu<double>(const cudaStream_t&, double*, const double*, const double*, const double*, const double*, int, int, int, bool, bool);


