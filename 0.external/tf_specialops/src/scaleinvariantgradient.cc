#define EIGEN_USE_GPU
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "cuda_helper.h"


using namespace tensorflow;

REGISTER_OP("ScaleInvariantGradient")
  .Attr("T: {float, double}")
  .Attr("deltas: list(int) = [1]")
  .Attr("weights: list(float) = [1.0]")
  .Attr("epsilon: float = 1e-3")
  .Input("input: T")
  .Output("output: T")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) 
    {
      using namespace ::tensorflow::shape_inference;
      ShapeHandle input_shape, output_shape;
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 2, &input_shape));

      if( c->RankKnown(input_shape) )
      {
        ShapeHandle shape2d;
        c->Subshape(input_shape, -2, &shape2d);

        ShapeHandle tmp_shape;
        c->Concatenate(c->MakeShape({2}), shape2d, &tmp_shape);


        int rank = c->Rank(input_shape);
        DimensionHandle first_dim = c->MakeDim(1);
        for( int i = 0; i < rank-2; ++i )
        {
          c->Multiply(first_dim, c->Dim(input_shape,i), &first_dim);
        }
        c->Concatenate(c->MakeShape({first_dim}), tmp_shape, &output_shape);
        c->set_output(0, output_shape);
      }
      else
      {
        c->set_output(0, c->UnknownShape());
      }
      return Status::OK();
    })
  .Doc(R"doc(
This op computes the scale invariant spatial gradient

The x component is computed as:
  grad_x = sum_delta w*(u(x+delta,y) - u(x,y))/(|u(x+delta,y)| + |u(x,y)| + eps)

Note that this op does not distinguish between channels and batch size of the 
input tensor. If the input tensor has more than one channel, then the resulting 
batch size will be the product of the input batch size and the channels.
E.g. (bi,ci,hi,wi) -> (bi*ci, 2, h, w).



input: 
  An input tensor with at least rank 2.

deltas:
  The pixel delta for the difference. 
  This vector must be the same length as weight.

weights:
  The weight factor for each difference.
  This vector must be the same length as delta.

epsilon:
  epsilon value for avoiding division by zero


output:=Tensor with the scale invariant spatial gradient.
  The format of the output tensor is NCHW with C=2; [batch, 2, height, width].
  The first channel is the x (width) component.
)doc");



template <class T>
void scaleinvariantgrad_gpu( 
    const cudaStream_t& stream,
    T* out, const T* in, 
    const int* deltas,
    const T* weights,
    int num_deltas,
    T epsilon,
    int x_size, int y_size, int z_size );


template <bool GPU, class T>
class ScaleInvariantGradientOp : public OpKernel 
{
public:
  explicit ScaleInvariantGradientOp(OpKernelConstruction* construction)
    :OpKernel(construction), persistent_initialized(false)
  { 
    std::vector<float> weights_tmp;
    float eps_tmp;
    OP_REQUIRES_OK(construction, construction->GetAttr("deltas", &deltas));
    OP_REQUIRES_OK(construction, construction->GetAttr("weights", &weights_tmp));
    OP_REQUIRES_OK(construction, construction->GetAttr("epsilon", &eps_tmp));
    epsilon = eps_tmp;
    for( float w : weights_tmp )
      weights.push_back(w);

    OP_REQUIRES(construction, deltas.size() == weights.size(),
        errors::InvalidArgument("The size of the deltas and weights vectors must be the same")
        );

  }

  void Compute( OpKernelContext* context ) override 
  {
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<T>();
    const TensorShape input_shape(input_tensor.shape());
    const int input_rank = input_shape.dims();

    TensorShape output_shape;
    int64_t w_size = 1;
    for( int i = 0; i < input_rank-2; ++i )
      w_size *= input_shape.dim_size(i);
    output_shape.AddDim(w_size);
    output_shape.AddDim(2);
    output_shape.AddDim(input_shape.dim_size(input_rank-2));
    output_shape.AddDim(input_shape.dim_size(input_rank-1));
    Tensor* output_tensor = 0;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));
    auto output = output_tensor->flat<T>();

    if(GPU)
    {
      auto device = context->eigen_gpu_device();
      Tensor* deltas_gpu_tensor;
      Tensor* weights_gpu_tensor;

      if( !persistent_initialized )
      {

        OP_REQUIRES_OK(context, context->allocate_persistent(
              DataTypeToEnum<int>::v(),
              TensorShape({int64_t(deltas.size())}),
              &deltas_gpu,
              &deltas_gpu_tensor ));
        int* deltas_gpu_ptr = deltas_gpu_tensor->flat<int>().data();

        //device.memcpyHostToDevice(deltas_gpu_ptr, &deltas[0], sizeof(T)*deltas.size());
        cudaMemcpyAsync(deltas_gpu_ptr, &deltas[0], sizeof(T)*deltas.size(), cudaMemcpyHostToDevice, device.stream());


        OP_REQUIRES_OK(context, context->allocate_persistent(
              DataTypeToEnum<T>::v(),
              TensorShape({int64_t(weights.size())}),
              &weights_gpu,
              &weights_gpu_tensor ));
        T* weights_gpu_ptr = weights_gpu_tensor->flat<T>().data();

        //device.memcpyHostToDevice(weights_gpu_ptr, &weights[0], sizeof(T)*weights.size());
        cudaMemcpyAsync(weights_gpu_ptr, &weights[0], sizeof(T)*weights.size(), cudaMemcpyHostToDevice, device.stream());
        persistent_initialized = true;
      }

      deltas_gpu_tensor = deltas_gpu.AccessTensor(context);
      weights_gpu_tensor = weights_gpu.AccessTensor(context);

      scaleinvariantgrad_gpu( 
          device.stream(),
          output.data(),
          input.data(),
          deltas_gpu_tensor->flat<int>().data(),
          weights_gpu_tensor->flat<T>().data(),
          deltas.size(),
          epsilon,
          input_shape.dim_size(input_rank-1),
          input_shape.dim_size(input_rank-2),
          w_size );
      
    }
    else
    {
      scaleinvariantgrad_cpu(
          output.data(),
          input.data(),
          input_shape.dim_size(input_rank-1),
          input_shape.dim_size(input_rank-2),
          w_size );
    }
    
  }

  void scaleinvariantgrad_cpu( 
      T* out, const T* in, 
      int x_size, int y_size, int z_size )
  {
    const int max_comparisons = deltas.size();
    const T eps = epsilon;
    const int xy_size = x_size*y_size;

    for( int z = 0; z < z_size; ++z )
    {
      T* out_z = out+2*z*xy_size;
      const T* in_z = in+z*xy_size;
#define IN(y,x) in_z[(y)*x_size+(x)]
#define OUT(c,y,x) out_z[(c)*xy_size+(y)*x_size+(x)]
      for( int y = 0; y < y_size; ++y )
      for( int x = 0; x < x_size; ++x )
      {
        const T value0 = IN(y,x);
        T grad_x = 0;
        T grad_y = 0;
        for( int comparison = 0; comparison < max_comparisons; ++comparison )
        {
          int delta = deltas[comparison];
          T weight = weights[comparison];

          T valuex, valuey;
          if( x+delta >= 0 && x+delta < x_size )
            valuex = IN(y,x+delta);
          else
            valuex = value0;

          if( y+delta >= 0 && y+delta < y_size )
            valuey = IN(y+delta,x);
          else
            valuey = value0;

          grad_x += weight*(valuex-value0)/(std::abs(value0)+std::abs(valuex)+eps);
          grad_y += weight*(valuey-value0)/(std::abs(value0)+std::abs(valuey)+eps);
        }
        OUT(0,y,x) = grad_x;
        OUT(1,y,x) = grad_y;
      }
#undef IN
#undef OUT

    } // z

  }



private:
  std::vector<int> deltas;
  std::vector<T> weights;
  T epsilon;
  PersistentTensor deltas_gpu;
  PersistentTensor weights_gpu;
  bool persistent_initialized;

};


#define REG_KB(type)                                                          \
REGISTER_KERNEL_BUILDER(                                                      \
    Name("ScaleInvariantGradient")                                            \
    .Device(DEVICE_CPU)                                                       \
    .TypeConstraint<type>("T"),                                               \
    ScaleInvariantGradientOp<false,type>);                                                
REG_KB(float)
REG_KB(double)
#undef REG_KB

#define REG_KB(type)                                                          \
REGISTER_KERNEL_BUILDER(                                                      \
    Name("ScaleInvariantGradient")                                            \
    .Device(DEVICE_GPU)                                                       \
    .TypeConstraint<type>("T"),                                               \
    ScaleInvariantGradientOp<true,type>);                                      
REG_KB(float)
REG_KB(double)
#undef REG_KB





REGISTER_OP("ScaleInvariantGradientGrad")
  .Attr("T: {float, double}")
  .Attr("deltas: list(int) = [1]")
  .Attr("weights: list(float) = [1.0]")
  .Attr("epsilon: float = 1e-3")
  .Input("gradients: T")
  .Input("input: T")
  .Output("backprops: T")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) 
    {
      c->set_output(0, c->input(1));
      return Status::OK();
    })
  .Doc(R"doc(
This op computes the gradient for ScaleInvariantGradient.
)doc");


namespace scaleinvariantgrad_internal
{

  template <class T>
  inline bool is_valid(const T& value)
  {
    return std::isfinite(value);
  }

  template <class T>
  inline T dcenter(const T& center_value, const T& neighbour_value, const T& eps)
  {
    T sum_abs = std::abs(center_value) + std::abs(neighbour_value) + eps;
    T sign = -1;
    if( center_value < 0 )
      sign = 1;
    return -1/sum_abs + sign*(neighbour_value - center_value)/(sum_abs*sum_abs);
  }

  template <class T>
  inline T dneighbour(const T& center_value, const T& neighbour_value, const T& eps)
  {
    T sum_abs = std::abs(center_value) + std::abs(neighbour_value) + eps;
    T sign = -1;
    if( neighbour_value < 0 )
      sign = 1;
    return 1/sum_abs + sign*(neighbour_value - center_value)/(sum_abs*sum_abs);
  }

} 
using namespace scaleinvariantgrad_internal;

template <class T>
void scaleinvariantgrad_grad_gpu( 
    const cudaStream_t& stream,
    T* out, const T* in, const T* grad, 
    const int* deltas, const T* weights, int num_deltas, T eps,
    int x_size, int y_size, int z_size );

template <bool GPU, class T>
class ScaleInvariantGradientGradOp : public OpKernel 
{
public:
  explicit ScaleInvariantGradientGradOp(OpKernelConstruction* construction)
    :OpKernel(construction), persistent_initialized(false)
  { 
    std::vector<float> weights_tmp;
    float eps_tmp;
    OP_REQUIRES_OK(construction, construction->GetAttr("deltas", &deltas));
    OP_REQUIRES_OK(construction, construction->GetAttr("weights", &weights_tmp));
    OP_REQUIRES_OK(construction, construction->GetAttr("epsilon", &eps_tmp));
    epsilon = eps_tmp;
    for( float w : weights_tmp )
      weights.push_back(w);

    OP_REQUIRES(construction, deltas.size() == weights.size(),
        errors::InvalidArgument("The size of the deltas and weights vectors must be the same")
        );

  }

  void Compute( OpKernelContext* context ) override 
  {
    const Tensor& gradient_tensor = context->input(0);
    auto gradient = gradient_tensor.flat<T>();
    const TensorShape gradient_shape(gradient_tensor.shape());
    
    const Tensor& input_tensor = context->input(1);
    auto input = input_tensor.flat<T>();
    const TensorShape input_shape(input_tensor.shape());
    const int input_rank = input_shape.dims();

    TensorShape output_shape(input_shape);
    int64_t w_size = 1;
    for( int i = 0; i < input_rank-2; ++i )
      w_size *= input_shape.dim_size(i);
    Tensor* output_tensor = 0;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));
    auto output = output_tensor->flat<T>();

    if(GPU)
    {
      auto device = context->eigen_gpu_device();
      Tensor* deltas_gpu_tensor;
      Tensor* weights_gpu_tensor;

      if( !persistent_initialized )
      {

        OP_REQUIRES_OK(context, context->allocate_persistent(
              DataTypeToEnum<int>::v(),
              TensorShape({int64_t(deltas.size())}),
              &deltas_gpu,
              &deltas_gpu_tensor ));
        int* deltas_gpu_ptr = deltas_gpu_tensor->flat<int>().data();

        cudaMemcpyAsync(deltas_gpu_ptr, &deltas[0], sizeof(T)*deltas.size(), cudaMemcpyHostToDevice, device.stream());

        OP_REQUIRES_OK(context, context->allocate_persistent(
              DataTypeToEnum<T>::v(),
              TensorShape({int64_t(weights.size())}),
              &weights_gpu,
              &weights_gpu_tensor ));
        T* weights_gpu_ptr = weights_gpu_tensor->flat<T>().data();

        cudaMemcpyAsync(weights_gpu_ptr, &weights[0], sizeof(T)*weights.size(), cudaMemcpyHostToDevice, device.stream());
        persistent_initialized = true;
      }

      deltas_gpu_tensor = deltas_gpu.AccessTensor(context);
      weights_gpu_tensor = weights_gpu.AccessTensor(context);

      scaleinvariantgrad_grad_gpu( 
          device.stream(),
          output.data(),
          input.data(),
          gradient.data(),
          deltas_gpu_tensor->flat<int>().data(),
          weights_gpu_tensor->flat<T>().data(),
          deltas.size(),
          epsilon,
          input_shape.dim_size(input_rank-1),
          input_shape.dim_size(input_rank-2),
          w_size );
    }
    else
    {
      scaleinvariantgrad_grad_cpu(
          output.data(),
          input.data(),
          gradient.data(),
          &deltas[0], &weights[0], deltas.size(), epsilon,
          input_shape.dim_size(input_rank-1),
          input_shape.dim_size(input_rank-2),
          w_size );
    }
    
  }

  void scaleinvariantgrad_grad_cpu( 
      T* out, const T* in, const T* grad, 
      const int* deltas, const T* weights, int num_deltas, T eps,
      int x_size, int y_size, int z_size )
  {
    const int xy_size = x_size*y_size;
    const int max_comparisons = num_deltas;

#define INPUT(x,y,z) in[(z)*xy_size+(y)*x_size+(x)]
#define OUT(x,y,z) out[(z)*xy_size+(y)*x_size+(x)]
#define GRAD(x,y,z) grad[(z)*xy_size+(y)*x_size+(x)]
    for( int z = 0; z < z_size; ++z )
    for( int y = 0; y < y_size; ++y )
    for( int x = 0; x < x_size; ++x )
    {
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

      if( !std::isfinite(value0_diff) )
        value0_diff = 0;
      OUT(x,y,z) = value0_diff;
    }
#undef INPUT
#undef OUT
#undef GRAD

  }



private:
  std::vector<int> deltas;
  std::vector<T> weights;
  T epsilon;
  PersistentTensor deltas_gpu;
  PersistentTensor weights_gpu;
  bool persistent_initialized;

};


#define REG_KB(type)                                                          \
REGISTER_KERNEL_BUILDER(                                                      \
    Name("ScaleInvariantGradientGrad")                                        \
    .Device(DEVICE_CPU)                                                       \
    .TypeConstraint<type>("T"),                                               \
    ScaleInvariantGradientGradOp<false,type>);                                 
REG_KB(float)
REG_KB(double)
#undef REG_KB

#define REG_KB(type)                                                          \
REGISTER_KERNEL_BUILDER(                                                      \
    Name("ScaleInvariantGradientGrad")                                        \
    .Device(DEVICE_GPU)                                                       \
    .TypeConstraint<type>("T"),                                               \
    ScaleInvariantGradientGradOp<true,type>);                                 
REG_KB(float)
REG_KB(double)
#undef REG_KB

