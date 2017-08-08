#define EIGEN_USE_GPU
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "cuda_helper.h"

using namespace tensorflow;

REGISTER_OP("Median3x3Downsample")
  .Attr("T: {float, double}")
  .Input("input: T")
  .Output("output: T")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) 
    {
      using namespace ::tensorflow::shape_inference;
      ShapeHandle input, output, tmp;
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 2, &input));
      if( c->RankKnown(input) )
      {
        DimensionHandle in_width_dim, in_height_dim;
        in_width_dim = c->Dim(input,-1);
        in_height_dim = c->Dim(input,-2);

        DimensionHandle out_width_dim, out_height_dim;
        if( c->ValueKnown(in_width_dim) )
          out_width_dim = c->MakeDim(divup(c->Value(in_width_dim),2));
        else
          out_width_dim = c->UnknownDim();

        if( c->ValueKnown(in_height_dim) )
          out_height_dim = c->MakeDim(divup(c->Value(in_height_dim),2));
        else
          out_height_dim = c->UnknownDim();

        c->ReplaceDim(input, -1, out_width_dim, &tmp);
        c->ReplaceDim(tmp, -2, out_height_dim, &output);
        c->set_output(0, output);
      }
      else // no rank information -> no shape information
      {
        c->set_output(0, c->UnknownShape());
      }
      return Status::OK();
    })
  .Doc(R"doc(
Downsamples an image with a 3x3 median filter with a stride of 2.
The input is at least a 2d tensor. 
The supported format is NCHW [batch, channels, height, width].
)doc");



template <class T>
void median3x3downsample_gpu(const cudaStream_t& stream, T* out, const T* in, int, int, int);


template <bool GPU, class T>
class Median3x3DownsampleOp : public OpKernel 
{
public:
  explicit Median3x3DownsampleOp(OpKernelConstruction* construction)
    :OpKernel(construction)
  { }

  void Compute( OpKernelContext* context ) override 
  {
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<T>();
    const TensorShape input_shape(input_tensor.shape());
    const int rank = input_shape.dims();
    TensorShape output_shape(input_tensor.shape());
    {
      int idx = rank-1;
      output_shape.set_dim(idx,divup(output_shape.dim_size(idx),2));
      idx = rank-2;
      output_shape.set_dim(idx,divup(output_shape.dim_size(idx),2));
    }
    Tensor* output_tensor = 0;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));
    auto output = output_tensor->flat<T>();

    int64_t z_size = 1;
    for( int i = 0; i < rank-2; ++i )
      z_size *= output_shape.dim_size(i);
    
    if( GPU )
    {
      auto device = context->eigen_gpu_device();
      median3x3downsample_gpu(
          device.stream(),
          output.data(), input.data(), 
          z_size, 
          input_shape.dim_size(rank-2),
          input_shape.dim_size(rank-1)
          );
    }
    else
    {
      median3x3downsample_cpu(
          output.data(), input.data(), 
          z_size, 
          input_shape.dim_size(rank-2),
          input_shape.dim_size(rank-1)
          );
    }
  }

  void median3x3downsample_cpu( 
      T* out, const T* in, 
      int z_size, 
      int in_y_size, int in_x_size )
  {
    T* out_ptr = out;
    for( int z = 0; z < z_size; ++z )
    for( int y = 0; y < in_y_size; y+=2 ) 
    for( int x = 0; x < in_x_size; x+=2 ) 
    {
      T value[9];
      int value_idx = 0;
      for( int dy = -1; dy <= 1; ++dy )
      for( int dx = -1; dx <= 1; ++dx )
      {
        int x_ = std::min(in_x_size-1,std::max(0,x+dx));
        int y_ = std::min(in_y_size-1,std::max(0,y+dy));
        value[value_idx++] = in[z*in_y_size*in_x_size+y_*in_x_size+x_];
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

      *out_ptr = value[4];
      ++out_ptr;
    }

  }

private:

};


#define REG_KB(type)                                                          \
REGISTER_KERNEL_BUILDER(                                                      \
    Name("Median3x3Downsample")                                               \
    .Device(DEVICE_CPU)                                                       \
    .TypeConstraint<type>("T"),                                               \
    Median3x3DownsampleOp<false,type>);                                        
REG_KB(float)
REG_KB(double)
#undef REG_KB

#define REG_KB(type)                                                          \
REGISTER_KERNEL_BUILDER(                                                      \
    Name("Median3x3Downsample")                                               \
    .Device(DEVICE_GPU)                                                       \
    .TypeConstraint<type>("T"),                                               \
    Median3x3DownsampleOp<true,type>);                                         
REG_KB(float)
REG_KB(double)
#undef REG_KB

